from rag.chunking import chunk_text
from rag.embeddings import HashingEmbeddingModel
from rag.retriever import FaissRetriever, RetrievedChunk


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=2)

    assert chunks == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]


def test_hashing_embeddings_are_normalized() -> None:
    model = HashingEmbeddingModel()

    vector = model.embed("vector search local context")

    assert vector.shape == (384,)
    assert 0.99 < float((vector**2).sum()) < 1.01


def test_faiss_retriever_returns_relevant_chunk() -> None:
    retriever = FaissRetriever()
    retriever.add(
        [
            RetrievedChunk("1", "doc", "apple banana fruit", 0),
            RetrievedChunk("2", "doc", "vector search retrieval", 0),
        ]
    )

    results = retriever.search("vector retrieval", top_k=1)

    assert results[0].chunk_id == "2"

