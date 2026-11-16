"""Small trainable heads that sit on top of frozen embeddings.

The embedder never changes. A head is a few thousand numbers trained on your
labelled vectors, so training takes milliseconds and inference is one matrix
multiply.

* ``ChoiceHead`` - pick one of N labels (softmax regression), with an
  optional "none of these" answer when confidence is low.
* ``ScoreHead``  - predict a number with an 80% interval (ridge regression
  plus split-conformal residuals).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ChoiceHead:
    """Multinomial logistic regression trained with full-batch gradient descent."""

    def __init__(self, labels: list[str], dim: int):
        self.labels = list(labels)
        self.W = np.zeros((dim, len(labels)), dtype=np.float32)
        self.b = np.zeros(len(labels), dtype=np.float32)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: list[str],
        *,
        epochs: int = 300,
        lr: float = 2.0,
        l2: float = 1e-3,
    ) -> "ChoiceHead":
        X = np.asarray(X, dtype=np.float32)
        labels = sorted(set(y))
        head = cls(labels, X.shape[1])
        idx = np.array([labels.index(v) for v in y])
        onehot = np.eye(len(labels), dtype=np.float32)[idx]
        n = len(X)
        for _ in range(epochs):
            p = _softmax(X @ head.W + head.b)
            grad = (p - onehot) / n
            head.W -= lr * (X.T @ grad + l2 * head.W)
            head.b -= lr * grad.sum(axis=0)
        return head

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return _softmax(X @ self.W + self.b)

    def predict(self, x: np.ndarray, none_below: float | None = None) -> tuple[str | None, dict[str, float]]:
        """Return (label, probabilities). Label is None when the top
        probability is under ``none_below`` ("none of these")."""
        p = self.predict_proba(x)[0]
        probs = {lab: float(v) for lab, v in zip(self.labels, p)}
        best = int(p.argmax())
        if none_below is not None and p[best] < none_below:
            return None, probs
        return self.labels[best], probs

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b, labels=np.array(self.labels))

    @classmethod
    def load(cls, path: str) -> "ChoiceHead":
        d = np.load(path)
        head = cls([str(v) for v in d["labels"]], d["W"].shape[0])
        head.W, head.b = d["W"], d["b"]
        return head


@dataclass
class ScorePrediction:
    value: float
    low: float
    high: float


class ScoreHead:
    """Ridge regression with a split-conformal prediction interval.

    Part of the training data is held out. The spread of the errors on that
    held-out part gives the interval: about ``coverage`` of future errors
    should fall inside it, provided new data looks like the training data.
    """

    def __init__(self, w: np.ndarray, b: float, half_width: float, coverage: float):
        self.w, self.b, self.half_width, self.coverage = w, b, half_width, coverage

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        l2: float = 0.1,
        coverage: float = 0.8,
        holdout: float = 0.25,
        seed: int = 0,
    ) -> "ScoreHead":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        if n < 8:
            raise ValueError("need at least 8 examples to fit a ScoreHead")
        perm = np.random.default_rng(seed).permutation(n)
        n_cal = max(2, int(round(n * holdout)))
        cal, train = perm[:n_cal], perm[n_cal:]

        def ridge(Xt, yt):
            mu_x, mu_y = Xt.mean(axis=0), yt.mean()
            Xc = Xt - mu_x
            # dual form: cheap when there are fewer examples than dimensions
            alpha = np.linalg.solve(Xc @ Xc.T + l2 * np.eye(len(Xt)), yt - mu_y)
            w = Xc.T @ alpha
            return w, float(mu_y - mu_x @ w)

        w, b = ridge(X[train], y[train])
        resid = np.sort(np.abs(y[cal] - (X[cal] @ w + b)))
        rank = min(len(resid) - 1, int(np.ceil((len(resid) + 1) * coverage)) - 1)
        half_width = float(resid[rank])
        w, b = ridge(X, y)  # refit on everything for the final weights
        return cls(w.astype(np.float32), b, half_width, coverage)

    def predict(self, x: np.ndarray) -> ScorePrediction:
        v = float(np.asarray(x, dtype=np.float32).reshape(-1) @ self.w + self.b)
        return ScorePrediction(v, v - self.half_width, v + self.half_width)
