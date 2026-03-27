from pathlib import Path
import json

from src.chunking.splitter import chunk_document_sections


INPUT_FILE = Path("data/processed/ingested_docs.json")
OUTPUT_FILE = Path("data/processed/chunks.json")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks = []

    for doc in docs:
        chunks = chunk_document_sections(doc, chunk_size=1500, overlap=200)
        all_chunks.extend(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()