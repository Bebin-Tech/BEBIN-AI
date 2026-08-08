from dataclasses import dataclass

import faiss
import numpy as np

from rag.embeddings import HashingEmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float


class Retriever:
    def add(self, chunks: list[RetrievedChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        raise NotImplementedError


class FaissRetriever(Retriever):
    def __init__(self, embedding_model: HashingEmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.index = faiss.IndexFlatIP(self.embedding_model.dimension)
        self.chunks: list[RetrievedChunk] = []

    def add(self, chunks: list[RetrievedChunk]) -> None:
        if not chunks:
            return
        vectors = self.embedding_model.embed_many([chunk.text for chunk in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if self.index.ntotal == 0 or top_k <= 0:
            return []

        query_vector = self.embedding_model.embed_many([query])
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

