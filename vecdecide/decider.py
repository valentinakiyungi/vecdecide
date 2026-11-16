"""The front door: ask typed questions about text and get typed answers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .embedders import Embedder, HashingEmbedder
from .heads import ChoiceHead, ScoreHead, ScorePrediction
from .store import Neighbor, OutcomeStore


@dataclass
class Decision:
    """The answer to a question.

    ``answer`` is a label for choice questions, True/False for yes/no
    questions, or None for "none of these". ``scores`` holds the per-option
    numbers behind it. ``evidence`` lists the past cases used, if any.
    """
    answer: str | bool | None
    confidence: float
    scores: dict[str, float]
    method: str
    evidence: list[Neighbor] = field(default_factory=list)


class Decider:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._cache: dict[str, np.ndarray] = {}

    # -- embedding (cached, so repeated questions over the same text are free)
    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed([text])[0]
        return self._cache[text]

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def _vec(self, item: str | np.ndarray) -> np.ndarray:
        return self.embed(item) if isinstance(item, str) else np.asarray(item, dtype=np.float32)

    # -- 1. zero-shot: no labelled data at all ----------------------------
    def choose(
        self,
        item: str | np.ndarray,
        options: Sequence[str] | dict[str, str],
        *,
        none_below: float | None = None,
        temperature: float = 0.1,
    ) -> Decision:
        """Pick the option whose description is most similar to the item.

        ``options`` is a list of labels, or a dict of label -> description.
        If ``none_below`` is set and the best cosine similarity is under it,
        the answer is None.
        """
        desc = dict(options) if isinstance(options, dict) else {o: o for o in options}
        labels = list(desc)
        sims = self.embed_many([desc[l] for l in labels]) @ self._vec(item)
        z = sims / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        best = int(sims.argmax())
        answer = labels[best]
        if none_below is not None and sims[best] < none_below:
            answer = None
        return Decision(answer, float(p[best]),
                        {l: float(s) for l, s in zip(labels, sims)}, "zero-shot")

    def yes_no(self, item: str | np.ndarray, yes: str, no: str) -> Decision:
        """Is the item closer to the ``yes`` description or the ``no`` one?"""
        d = self.choose(item, {"yes": yes, "no": no})
        return Decision(d.answer == "yes", d.confidence, d.scores, "zero-shot")

    # -- 2. outcome store: decide from what worked before -----------------
    def remember(self, store: OutcomeStore, text: str, option: str,
                 outcome: float = 1.0, **meta) -> int:
        """Write a case and its outcome back to the store."""
        return store.add(self.embed(text), option, outcome, text=text, meta=meta)

    def decide_from_outcomes(
        self,
        item: str | np.ndarray,
        store: OutcomeStore,
        *,
        options: Sequence[str] | None = None,
        k: int = 8,
        min_similarity: float = 0.05,
        prior_outcome: float = 0.0,
        prior_weight: float = 0.5,
    ) -> Decision:
        """Pick the option whose similar past cases had the best outcomes.

        Each of the k nearest cases votes for its option, weighted by
        similarity. An option's score is the weighted mean outcome, pulled
        slightly toward ``prior_outcome`` so one lucky case does not win
        outright. If no past case is at least ``min_similarity`` close,
        the answer is None.
        """
        vec = self._vec(item)
        near = [n for n in store.nearest(vec, k) if n.similarity >= min_similarity]
        candidates = list(options) if options is not None else sorted(set(store.options))
        scores = {o: prior_outcome for o in candidates}
        support = {o: 0.0 for o in candidates}
        for o in candidates:
            mine = [n for n in near if n.option == o]
            w = sum(n.similarity for n in mine)
            support[o] = w
            scores[o] = (sum(n.similarity * n.outcome for n in mine)
                         + prior_weight * prior_outcome) / (w + prior_weight)
        if not near or not candidates:
            return Decision(None, 0.0, scores, "outcome-store", near)
        best = max(candidates, key=lambda o: (scores[o], support[o]))
        total = sum(support.values())
        conf = support[best] / total if total > 0 else 0.0
        return Decision(best, float(conf), scores, "outcome-store", near)

    # -- 3. trained heads --------------------------------------------------
    def train_choice_head(self, texts: Sequence[str], labels: Sequence[str], **kw) -> ChoiceHead:
        return ChoiceHead.fit(self.embed_many(texts), list(labels), **kw)

    def choose_with_head(self, item: str | np.ndarray, head: ChoiceHead,
                         *, none_below: float | None = None) -> Decision:
        label, probs = head.predict(self._vec(item), none_below)
        return Decision(label, max(probs.values()), probs, "choice-head")

    def compile_store(self, store: OutcomeStore, *, min_outcome: float = 0.5, **kw) -> ChoiceHead:
        """Distil an outcome store into a ChoiceHead.

        Keeps only the cases that went well (outcome >= ``min_outcome``) and
        trains a head to predict their option. The head is then a fixed-size
        stand-in for the store: no neighbour search at decision time.
        """
        keep = store.outcomes >= min_outcome
        if not keep.any():
            raise ValueError("no cases with a good enough outcome to learn from")
        labels = [o for o, k in zip(store.options, keep) if k]
        return ChoiceHead.fit(store.vectors[keep], labels, **kw)

    def train_score_head(self, texts: Sequence[str], values: Sequence[float], **kw) -> ScoreHead:
        return ScoreHead.fit(self.embed_many(texts), np.asarray(values, dtype=float), **kw)

    def score(self, item: str | np.ndarray, head: ScoreHead) -> ScorePrediction:
        return head.predict(self._vec(item))
