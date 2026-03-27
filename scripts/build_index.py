from pathlib import Path
import json

from src.embeddings.embedder import LocalEmbedder
from src.retrieval.vector_store import ChromaVectorStore


CHUNKS_FILE = Path("data/processed/chunks.json")


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("No chunks found in data/processed/chunks.json")

    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])

        enriched_text = (
            f"Source: {chunk['source_name']}\n"
            f"Section: {chunk['section_title']}\n"
            f"Content: {chunk['text']}"
        )
        texts.append(enriched_text)

        metadatas.append(
            {
                "doc_id": chunk["doc_id"],
                "source_name": chunk["source_name"],
                "source_path": chunk["source_path"],
                "section_title": chunk["section_title"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "raw_text": chunk["text"],
            }
        )

    print(f"ids count       : {len(ids)}")
    print(f"texts count     : {len(texts)}")
    print(f"metadatas count : {len(metadatas)}")

    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed_texts(texts)

    print(f"embeddings count: {len(embeddings)}")

    vector_store = ChromaVectorStore(path="chroma_db", collection_name="httpx_docs")
    vector_store.add_chunks(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(chunks)} chunks into Chroma collection 'httpx_docs'")


if __name__ == "__main__":
    main()