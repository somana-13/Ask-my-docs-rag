from src.retrieval.bm25_retriever import LocalBM25Retriever


def main():
    print("Starting BM25 retrieval...")
    print("Type 'exit' to quit.\n")

    retriever = LocalBM25Retriever(k=5)

    while True:
        query = input("Enter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = retriever.query(query)

        print("\n" + "=" * 100)
        print(f"QUESTION: {query}")
        print("=" * 100)

        for i, doc in enumerate(results, start=1):
            meta = doc.metadata
            print(f"\nResult #{i}")
            print(f"Source       : {meta.get('source_name')}")
            print(f"Section      : {meta.get('section_title')}")
            print(f"Chunk ID     : {meta.get('chunk_id')}")
            print("\nEvidence:\n")
            print(meta.get("raw_text", "")[:1000])
            print("\n" + "-" * 100)


if __name__ == "__main__":
    main()