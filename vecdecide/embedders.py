"""Embedders turn text into fixed-size unit vectors.

Anything with an ``embed(texts) -> np.ndarray`` method and a ``dim`` attribute
works. Two are provided:

* ``HashingEmbedder`` - no downloads, no dependencies beyond numpy. It hashes
  words and character n-grams into a fixed number of buckets. It captures
  lexical overlap, not meaning, so it is a baseline and a test double.
* ``SentenceTransformerEmbedder`` - wraps a real neural embedding model
  (requires ``pip install sentence-transformers``). Use this for real work.
"""
from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")

# Very common English words carry little signal and make unrelated texts look
# alike, so the hashing embedder skips them.
STOPWORDS = frozenset("""
a an the and or but if so of to in on at by for from with about into over after before
than then as is are was were be been being am do does did done have has had having
i me my mine we us our you your he him his she her it its they them their this that these
those there here what which who whom whose when where why how all any both each few more
most other some such no nor not only own same too very can will just should would could
may might must shall please also get got still since yet up out off again once
""".split())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalised rows."""
        ...


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows; all-zero rows stay zero."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class HashingEmbedder:
    """Signed feature hashing over words, word bigrams and character n-grams."""

    def __init__(self, dim: int = 1024, char_ngrams: tuple[int, ...] = (3, 4),
                 stopwords: frozenset[str] = STOPWORDS):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.stopwords = stopwords

    def _features(self, text: str):
        words = [w for w in _WORD.findall(text.lower()) if w not in self.stopwords]
        for w in words:
            yield "w:" + w, 1.0
            for n in self.char_ngrams:
                padded = f"<{w}>"
                for i in range(len(padded) - n + 1):
                    yield "c:" + padded[i : i + n], 0.5
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}", 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))  # stable across runs
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return normalize(out)


class SentenceTransformerEmbedder:
    """A real semantic embedder. The model is downloaded on first use."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional dependency: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
