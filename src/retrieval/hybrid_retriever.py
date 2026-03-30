from src.embeddings.embedder import LocalEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import LocalBM25Retriever


class HybridRetriever:
    def __init__(
        self,
        chroma_path: str = "chroma_db",
        collection_name: str = "httpx_docs",
        chunks_file: str = "data/processed/chunks.json",
        dense_k: int = 5,
        bm25_k: int = 5,
    ):
        self.embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        self.vector_store = ChromaVectorStore(path=chroma_path, collection_name=collection_name)
        self.bm25 = LocalBM25Retriever(chunks_file=chunks_file, k=bm25_k)
        self.dense_k = dense_k

    def dense_query(self, query: str):
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(query_embedding=query_embedding, n_results=self.dense_k)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        dense_results = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            dense_results.append(
                {
                    "chunk_id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                    "dense_distance": dist,
                    "retrieval_source": "dense",
                }
            )
        return dense_results

    def bm25_query(self, query: str):
        docs = self.bm25.query(query)

        bm25_results = []
        for doc in docs:
            meta = doc.metadata
            bm25_results.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "document": doc.page_content,
                    "metadata": meta,
                    "dense_distance": None,
                    "retrieval_source": "bm25",
                }
            )
        return bm25_results

    def query(self, query: str):
        dense_results = self.dense_query(query)
        bm25_results = self.bm25_query(query)

        merged = {}
        for item in dense_results + bm25_results:
            chunk_id = item["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = item
            else:
                existing_source = merged[chunk_id]["retrieval_source"]
                if existing_source != item["retrieval_source"]:
                    merged[chunk_id]["retrieval_source"] = "dense+bm25"

        return list(merged.values())