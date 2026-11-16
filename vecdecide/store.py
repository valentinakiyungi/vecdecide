"""An outcome store: past cases, what was chosen, and how well it went.

Each record is (vector, option, outcome, optional text/metadata). To decide a
new case we look up its nearest past cases and prefer the option that worked
best for them. No training step: writing a new outcome changes future answers
immediately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedders import normalize


@dataclass
class Neighbor:
    index: int
    similarity: float
    option: str
    outcome: float
    text: str | None
    meta: dict[str, Any]


class OutcomeStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._options: list[str] = []
        self._outcomes: list[float] = []
        self._texts: list[str | None] = []
        self._meta: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._options)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @property
    def options(self) -> list[str]:
        return list(self._options)

    @property
    def outcomes(self) -> np.ndarray:
        return np.asarray(self._outcomes, dtype=np.float32)

    def add(
        self,
        vector: np.ndarray,
        option: str,
        outcome: float = 1.0,
        text: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Record that ``option`` was chosen for this case with this outcome.

        ``outcome`` is any number where higher is better (1/0 for
        success/failure, a rating, a reward...). Returns the record index.
        """
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"expected a vector of size {self.dim}, got {vector.shape[0]}")
        self._vectors = np.vstack([self._vectors, normalize(vector)[None, :]])
        self._options.append(option)
        self._outcomes.append(float(outcome))
        self._texts.append(text)
        self._meta.append(dict(meta or {}))
        return len(self) - 1

    def nearest(self, vector: np.ndarray, k: int = 8) -> list[Neighbor]:
        """The k most similar past cases by cosine similarity (brute force)."""
        if len(self) == 0:
            return []
        q = normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        sims = self._vectors @ q
        k = min(k, len(self))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            Neighbor(int(i), float(sims[i]), self._options[i], self._outcomes[i],
                     self._texts[i], self._meta[i])
            for i in top
        ]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write ``<path>.npz`` (vectors) and ``<path>.json`` (records)."""
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), vectors=self._vectors)
        path.with_suffix(".json").write_text(json.dumps({
            "dim": self.dim,
            "options": self._options,
            "outcomes": self._outcomes,
            "texts": self._texts,
            "meta": self._meta,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "OutcomeStore":
        path = Path(path)
        data = json.loads(path.with_suffix(".json").read_text())
        store = cls(data["dim"])
        store._vectors = np.load(path.with_suffix(".npz"))["vectors"].astype(np.float32)
        store._options = data["options"]
        store._outcomes = data["outcomes"]
        store._texts = data["texts"]
        store._meta = data["meta"]
        return store
