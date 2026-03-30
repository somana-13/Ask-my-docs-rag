from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    print("Starting hybrid retrieval...")
    print("Type 'exit' to quit.\n")

    retriever = HybridRetriever(dense_k=5, bm25_k=5)

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

        for i, item in enumerate(results[:8], start=1):
            meta = item["metadata"]
            raw_text = meta.get("raw_text", "")[:900]

            print(f"\nResult #{i}")
            print(f"Source          : {meta.get('source_name')}")
            print(f"Section         : {meta.get('section_title')}")
            print(f"Chunk ID        : {item.get('chunk_id')}")
            print(f"Retrieved By    : {item.get('retrieval_source')}")
            print(f"Dense Distance  : {item.get('dense_distance')}")
            print("\nEvidence:\n")
            print(raw_text)
            print("\n" + "-" * 100)


if __name__ == "__main__":
    main()