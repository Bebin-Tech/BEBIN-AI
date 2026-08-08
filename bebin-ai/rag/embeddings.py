from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class HashingEmbeddingConfig:
    dimension: int = 384


class HashingEmbeddingModel:
    """Local deterministic embedding model used until neural embeddings are trained."""

    def __init__(self, config: HashingEmbeddingConfig | None = None) -> None:
        self.config = config or HashingEmbeddingConfig()
        if self.config.dimension <= 0:
            raise ValueError("embedding dimension must be positive")

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)


def _tokens(text: str) -> list[str]:
    return [token for token in text.lower().replace("\n", " ").split(" ") if token]

