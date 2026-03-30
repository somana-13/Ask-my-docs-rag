from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LocalReranker


MAX_CANDIDATES = 8
TOP_K = 5
MIN_RERANK_SCORE = 0.0
ABSTAIN_TOP_SCORE = 1.5


def filter_by_rerank_score(results, min_score=MIN_RERANK_SCORE):
    return [item for item in results if item.get("rerank_score", -999) >= min_score]


def should_abstain(results, top_score_threshold=ABSTAIN_TOP_SCORE):
    if not results:
        return True

    top_score = results[0].get("rerank_score", -999)
    return top_score < top_score_threshold


def main():
    print("Starting hybrid retrieval + reranking...")
    print("Type 'exit' to quit.\n")

    retriever = HybridRetriever(dense_k=5, bm25_k=5)
    reranker = LocalReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    while True:
        query = input("Enter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query:
            print("Please enter a question.")
            continue

        candidates = retriever.query(query)
        candidates = candidates[:MAX_CANDIDATES]
        reranked = reranker.rerank(query, candidates, top_k=TOP_K)
        reranked = filter_by_rerank_score(reranked)

        print("\n" + "=" * 100)
        print(f"QUESTION: {query}")
        print("=" * 100)

        if should_abstain(reranked):
            print("\nI could not find strong enough supporting evidence in the indexed documents.")
            print("Please rephrase the question or ask about a topic covered in the corpus.")
            continue

        for i, item in enumerate(reranked, start=1):
            meta = item["metadata"]
            raw_text = meta.get("raw_text", "")[:900]

            print(f"\nResult #{i}")
            print(f"Source          : {meta.get('source_name')}")
            print(f"Section         : {meta.get('section_title')}")
            print(f"Chunk ID        : {item.get('chunk_id')}")
            print(f"Retrieved By    : {item.get('retrieval_source')}")
            print(f"Dense Distance  : {item.get('dense_distance')}")
            print(f"Rerank Score    : {item.get('rerank_score'):.4f}")
            print("\nEvidence:\n")
            print(raw_text)
            print("\n" + "-" * 100)

        print("\nCITED SOURCES:")
        seen = set()
        for item in reranked:
            meta = item["metadata"]
            key = (meta.get("source_name"), meta.get("section_title"))
            if key not in seen:
                seen.add(key)
                print(f"- {meta.get('source_name')} | {meta.get('section_title')}")


if __name__ == "__main__":
    main()