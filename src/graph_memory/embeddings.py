"""Offline, deterministic embeddings so the demo runs without an API key.

Feature hashing over word tokens and character trigrams. It captures lexical
overlap only, not meaning. Swap `HashingEmbedder` for a real model (OpenAI,
Voyage, sentence-transformers…) with the same `embed()` signature and set
EMBEDDING_DIM to match.
"""

import hashlib
import math
import re
from typing import Protocol

_STOP = {"a", "an", "the", "to", "of", "at", "in", "on", "is", "my", "i", "and", "for", "with"}


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
        for w in words:
            i, sign = self._bucket("w:" + w)
            vec[i] += 2.0 * sign
            padded = f"#{w}#"
            for j in range(len(padded) - 2):
                i, sign = self._bucket("c:" + padded[j : j + 3])
                vec[i] += 0.5 * sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class LocalEmbedder:
    """Semantic embeddings from a local sentence-transformers model (no API).

    The default model (all-MiniLM-L6-v2) has 384 dimensions, so it matches the
    default EMBEDDING_DIM. A different model needs a matching EMBEDDING_DIM.
    """

    def __init__(self, dim: int = 384, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        actual = self.model.get_sentence_embedding_dimension()
        if actual != dim:
            raise ValueError(f"{model} produces {actual}-dim vectors; set EMBEDDING_DIM={actual}")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.dim
        return self.model.encode(text, normalize_embeddings=True).tolist()
