"""A saved model folder: an outcome store, a trained head and their settings.

Layout of a model folder::

    my-model/
        config.json   which embedder, which CSV columns, labels, counts
        store.npz     the vectors of every recorded case
        store.json    the label, outcome and text of every recorded case
        head.npz      a ChoiceHead trained from the good outcomes (optional)

Everything here works from plain CSV files, so you can bring your own data.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .decider import Decider, Decision
from .embedders import Embedder, HashingEmbedder
from .heads import ChoiceHead
from .store import OutcomeStore

DEFAULT_EMBEDDER = "hashing"


def default_min_similarity(spec: str) -> float:
    """Below this cosine similarity a past case counts as unrelated.

    Word-hashing vectors of unrelated texts sit near 0; sentence-transformer
    vectors of unrelated texts are usually 0.1-0.25, so they need a higher bar.
    These are starting points: tune them with ``vecdecide evaluate``.
    """
    return 0.10 if spec.split(":")[0] == "hashing" else 0.35


# -- embedders --------------------------------------------------------------
def make_embedder(spec: str) -> Embedder:
    """Build an embedder from a short spec string.

    * ``hashing`` or ``hashing:2048`` - the built-in word-matching baseline.
    * ``st`` - sentence-transformers' all-MiniLM-L6-v2.
    * ``st:<model name>`` - any sentence-transformers model, e.g.
      ``st:BAAI/bge-small-en-v1.5``.
    """
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(dim=int(arg) if arg else 1024)
    if kind in ("st", "sentence-transformers"):
        from .embedders import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(arg) if arg else SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {spec!r}; use 'hashing' or 'st:<model>'")


# -- CSV ---------------------------------------------------------------------
@dataclass
class Row:
    text: str
    label: str
    outcome: float = 1.0
    meta: dict = field(default_factory=dict)


def read_rows(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    outcome_col: str | None = "outcome",
) -> list[Row]:
    """Read labelled rows from a CSV file.

    The text and label columns are required. The outcome column is optional:
    if it is missing from the file every row counts as a good outcome (1.0).
    Any other columns are kept as metadata. Rows with empty text or label are
    skipped.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for needed in (text_col, label_col):
            if needed not in cols:
                raise ValueError(
                    f"{path.name} has no column {needed!r}. Columns found: {', '.join(cols)}. "
                    f"Use --text-col / --label-col to pick the right ones."
                )
        has_outcome = outcome_col is not None and outcome_col in cols
        rows = []
        for i, rec in enumerate(reader, start=2):
            text = (rec.get(text_col) or "").strip()
            label = (rec.get(label_col) or "").strip()
            if not text or not label:
                continue
            outcome = 1.0
            if has_outcome and (rec.get(outcome_col) or "").strip():
                try:
                    outcome = float(rec[outcome_col])
                except ValueError:
                    raise ValueError(
                        f"{path.name} line {i}: outcome {rec[outcome_col]!r} is not a number"
                    ) from None
            meta = {k: v for k, v in rec.items()
                    if k not in (text_col, label_col, outcome_col) and k is not None}
            rows.append(Row(text, label, outcome, meta))
    if not rows:
        raise ValueError(f"{path.name} has no usable rows")
    return rows


def read_texts(path: str | Path, text_col: str = "text") -> tuple[list[dict], list[str]]:
    """Read every row of a CSV and return (records, texts) for batch decisions."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if text_col not in (reader.fieldnames or []):
            raise ValueError(
                f"{path.name} has no column {text_col!r}. "
                f"Columns found: {', '.join(reader.fieldnames or [])}."
            )
        records = list(reader)
    return records, [(r.get(text_col) or "").strip() for r in records]


# -- the model folder --------------------------------------------------------
class Model:
    """An outcome store plus an optional trained head, saved in one folder."""

    def __init__(self, decider: Decider, store: OutcomeStore,
                 head: ChoiceHead | None = None, config: dict | None = None):
        self.decider = decider
        self.store = store
        self.head = head
        self.config = dict(config or {})

    # building
    @classmethod
    def build(cls, rows: Iterable[Row], embedder: str = DEFAULT_EMBEDDER,
              train_head: bool = True, min_outcome: float = 0.5,
              decider: Decider | None = None) -> "Model":
        rows = list(rows)
        decider = decider or Decider(make_embedder(embedder))
        store = OutcomeStore(decider.embedder.dim)
        vecs = decider.embed_many([r.text for r in rows])
        for r, v in zip(rows, vecs):
            store.add(v, r.label, r.outcome, text=r.text, meta=r.meta)
        model = cls(decider, store, config={
            "embedder": embedder,
            "dim": decider.embedder.dim,
            "min_outcome": min_outcome,
            "min_similarity": default_min_similarity(embedder),
        })
        if train_head:
            model.retrain()
        return model

    def retrain(self) -> ChoiceHead | None:
        """Re-fit the head from the store's good outcomes (needs 2+ labels)."""
        good = {o for o, v in zip(self.store.options, self.store.outcomes)
                if v >= self.config.get("min_outcome", 0.5)}
        self.head = (self.decider.compile_store(self.store, min_outcome=self.config.get("min_outcome", 0.5))
                     if len(good) >= 2 else None)
        return self.head

    @property
    def labels(self) -> list[str]:
        return sorted(set(self.store.options))

    # saving
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.store.save(folder / "store")
        head_path = folder / "head.npz"
        if self.head is not None:
            self.head.save(str(head_path))
        elif head_path.exists():
            head_path.unlink()
        cfg = dict(self.config, labels=self.labels, cases=len(self.store),
                   has_head=self.head is not None)
        (folder / "config.json").write_text(json.dumps(cfg, indent=2))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Model":
        folder = Path(folder)
        cfg_path = folder / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{folder} is not a vecdecide model folder (no config.json). "
                f"Create one with: vecdecide build data.csv -o {folder}"
            )
        cfg = json.loads(cfg_path.read_text())
        decider = Decider(make_embedder(cfg["embedder"]))
        if decider.embedder.dim != cfg["dim"]:
            raise ValueError("the embedder no longer matches the one this model was built with")
        store = OutcomeStore.load(folder / "store")
        head = ChoiceHead.load(str(folder / "head.npz")) if (folder / "head.npz").exists() else None
        return cls(decider, store, head, cfg)

    # deciding
    def decide(self, text: str, method: str = "store", *, k: int = 8,
               min_similarity: float | None = None, none_below: float | None = None) -> Decision:
        """``method`` is "store" (nearest past outcomes) or "head" (trained head)."""
        if method == "head":
            if self.head is None:
                raise ValueError("this model has no trained head; it needs at least two labels "
                                 "with good outcomes")
            return self.decider.choose_with_head(text, self.head, none_below=none_below)
        if method == "store":
            if min_similarity is None:
                min_similarity = self.config.get(
                    "min_similarity", default_min_similarity(self.config.get("embedder", "hashing")))
            return self.decider.decide_from_outcomes(
                text, self.store, k=k, min_similarity=min_similarity)
        raise ValueError(f"unknown method {method!r}; use 'store' or 'head'")

    def record(self, text: str, label: str, outcome: float = 1.0, **meta) -> int:
        return self.decider.remember(self.store, text, label, outcome, **meta)


# -- evaluation --------------------------------------------------------------
@dataclass
class EvalResult:
    method: str
    accuracy: float
    answered: int
    total: int
    per_label: dict[str, tuple[int, int]]  # label -> (correct, total)
    confusion: dict[tuple[str, str], int]  # (true, predicted) -> count


def evaluate(rows: list[Row], embedder: str = DEFAULT_EMBEDDER, *, folds: int = 5,
             seed: int = 0, k: int = 8, min_outcome: float = 0.5,
             min_similarity: float | None = None) -> tuple[list[EvalResult], int]:
    """Cross-validate: every good-outcome row is held out once and predicted
    by a model built from all the other rows.

    Only rows with good outcomes are used as test questions, because for
    those we know the right answer. Bad-outcome rows always stay in the
    training side, since they teach the store what not to do. Each text is
    embedded once. Returns (results, number of test questions).
    """
    good_idx = [i for i, r in enumerate(rows) if r.outcome >= min_outcome]
    if len(good_idx) < 10:
        raise ValueError("need at least 10 rows with good outcomes to evaluate")
    if len({rows[i].label for i in good_idx}) < 2:
        raise ValueError("need at least two different labels to evaluate")
    rng = random.Random(seed)
    # stratified folds: deal each label's rows round-robin into the folds
    fold_of: dict[int, int] = {}
    by_label: dict[str, list[int]] = {}
    for i in good_idx:
        by_label.setdefault(rows[i].label, []).append(i)
    offset = 0
    for label in sorted(by_label):
        idxs = by_label[label]
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            fold_of[i] = (j + offset) % folds
        offset += len(idxs)

    decider = Decider(make_embedder(embedder))
    decider.embed_many([r.text for r in rows])  # embed everything once

    methods = ["baseline", "store", "head"]
    tallies = {m: {"ok": 0, "answered": 0, "per": {}, "conf": {}} for m in methods}
    majority_names: set[str] = set()
    n_test = 0
    for f in range(folds):
        test = [i for i in good_idx if fold_of[i] == f]
        if not test:
            continue
        test_set = set(test)
        train_rows = [r for i, r in enumerate(rows) if i not in test_set]
        model = Model.build(train_rows, embedder, min_outcome=min_outcome, decider=decider)
        counts: dict[str, int] = {}
        for r in train_rows:
            if r.outcome >= min_outcome:
                counts[r.label] = counts.get(r.label, 0) + 1
        majority = max(sorted(counts), key=counts.get)
        majority_names.add(majority)
        for i in test:
            r = rows[i]
            n_test += 1
            preds = {"baseline": majority,
                     "store": model.decide(r.text, "store", k=k,
                                           min_similarity=min_similarity).answer}
            if model.head is not None:
                preds["head"] = model.decide(r.text, "head").answer
            for m, pred in preds.items():
                t = tallies[m]
                pred_s = "(none)" if pred is None else str(pred)
                ok = pred_s == r.label
                t["ok"] += ok
                t["answered"] += pred is not None
                c = t["per"].setdefault(r.label, [0, 0])
                c[0] += ok
                c[1] += 1
                t["conf"][(r.label, pred_s)] = t["conf"].get((r.label, pred_s), 0) + 1

    results = []
    for m in methods:
        t = tallies[m]
        total = sum(c[1] for c in t["per"].values())
        if not total:
            continue
        name = (f"baseline (always '{next(iter(majority_names))}')"
                if m == "baseline" and len(majority_names) == 1
                else "baseline (most common label)" if m == "baseline" else m)
        results.append(EvalResult(name, t["ok"] / total, t["answered"], total,
                                  {} if m == "baseline" else {k_: (v[0], v[1]) for k_, v in t["per"].items()},
                                  {} if m == "baseline" else t["conf"]))
    return results, n_test
