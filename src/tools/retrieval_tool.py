from __future__ import annotations

from functools import lru_cache
from typing import Any

from scripts.query_index import run_retrieval
from src.embeddings.embedder import LocalEmbedder
from src.retrieval.vector_store import ChromaVectorStore


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "httpx_docs"


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """Load the embedding model once and reuse it across retrieval calls."""
    return LocalEmbedder(model_name=DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Connect to the existing Chroma vector store once and reuse it."""
    return ChromaVectorStore(
        path=DEFAULT_CHROMA_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
    )


def retrieve_documents(query: str) -> dict[str, Any]:
    """
    Agent-facing retrieval tool.

    This wraps the existing dense retrieval pipeline and returns structured
    evidence that LangGraph nodes can pass through the agent state.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        return {
            "query": query,
            "has_evidence": False,
            "evidence": [],
            "sources": [],
            "error": "Empty query.",
        }

    embedder = get_embedder()
    vector_store = get_vector_store()

    return run_retrieval(
        query=cleaned_query,
        embedder=embedder,
        vector_store=vector_store,
    )


if __name__ == "__main__":
    sample_query = "How do I configure authentication in HTTPX?"
    result = retrieve_documents(sample_query)

    print(f"Query: {result['query']}")
    print(f"Has evidence: {result['has_evidence']}")
    print("Sources:")
    for source in result["sources"]:
        print(source)

    if result["evidence"]:
        print("\nTop evidence snippet:")
        print(result["evidence"][0]["raw_text"][:500])
