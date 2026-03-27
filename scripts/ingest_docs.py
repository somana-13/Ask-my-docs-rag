from pathlib import Path
import json

from src.ingestion.loaders import load_markdown_document


RAW_DIR = Path("data/raw/httpx")
OUTPUT_FILE = Path("data/processed/ingested_docs.json")


def main():
    all_docs = []

    for file_path in sorted(RAW_DIR.rglob("*.md")):
        doc = load_markdown_document(str(file_path))
        all_docs.append(doc)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_docs)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()