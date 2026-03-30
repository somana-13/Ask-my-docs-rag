import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def simple_tokenizer(text: str) -> list[str]:
    """
    Normalize technical text for BM25:
    - lowercase
    - remove most punctuation
    - keep alphanumeric and underscore/dot words
    """
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_\.]+", text)


class LocalBM25Retriever:
    def __init__(self, chunks_file: str = "data/processed/chunks.json", k: int = 5):
        self.chunks_file = Path(chunks_file)
        self.k = k
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        docs = []
        for chunk in chunks:
            page_content = (
                f"Source: {chunk['source_name']}\n"
                f"Section: {chunk['section_title']}\n"
                f"Content: {chunk['text']}"
            )

            metadata = {
                "doc_id": chunk["doc_id"],
                "source_name": chunk["source_name"],
                "source_path": chunk["source_path"],
                "section_title": chunk["section_title"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "raw_text": chunk["text"],
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

        retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=simple_tokenizer,
        )
        retriever.k = self.k
        return retriever

    def query(self, query: str):
        return self.retriever.invoke(query)