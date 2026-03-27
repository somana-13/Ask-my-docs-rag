from src.embeddings.embedder import LocalEmbedder
from src.retrieval.vector_store import ChromaVectorStore


DISTANCE_THRESHOLD = 0.82
MAX_RESULTS = 5
MAX_DISPLAY = 3


def keep_relevant_results(ids, documents, metadatas, distances, threshold=DISTANCE_THRESHOLD):
    filtered = []

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        if dist <= threshold:
            filtered.append((chunk_id, doc, meta, dist))

    return filtered


def diversify_results(results):
    diversified = []
    seen = set()

    for item in results:
        _, _, meta, _ = item
        key = (meta.get("source_name"), meta.get("section_title"))

        if key not in seen:
            diversified.append(item)
            seen.add(key)

    return diversified


def format_sources(metadatas):
    seen = set()
    lines = []

    for meta in metadatas:
        source_name = meta.get("source_name", "unknown")
        section_title = meta.get("section_title", "unknown")
        key = (source_name, section_title)

        if key not in seen:
            seen.add(key)
            lines.append(f"- {source_name} | {section_title}")

    return lines


def format_evidence_summary(metadatas, max_points=3):
    summary_points = []

    for meta in metadatas[:max_points]:
        source_name = meta.get("source_name", "unknown")
        section_title = meta.get("section_title", "unknown")
        raw_text = meta.get("raw_text", "").strip().replace("\n", " ")
        snippet = raw_text[:220].strip()

        summary_points.append(
            f"- [{source_name} | {section_title}] {snippet}..."
        )

    return summary_points


def run_query(query, embedder, vector_store):
    query_embedding = embedder.embed_query(query)
    results = vector_store.query(query_embedding=query_embedding, n_results=MAX_RESULTS)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered = keep_relevant_results(ids, documents, metadatas, distances)
    filtered = diversify_results(filtered)

    print("\n" + "=" * 100)
    print(f"QUESTION: {query}")
    print("=" * 100)

    if not filtered:
        print("\nNo strong supporting evidence was found in the indexed documents.")
        return

    ids = [x[0] for x in filtered][:MAX_DISPLAY]
    metadatas = [x[2] for x in filtered][:MAX_DISPLAY]
    distances = [x[3] for x in filtered][:MAX_DISPLAY]

    print("\nTOP RETRIEVED EVIDENCE:")
    for i, (chunk_id, meta, dist) in enumerate(zip(ids, metadatas, distances), start=1):
        print(
            f"{i}. {meta.get('source_name')} | {meta.get('section_title')} "
            f"(chunk_id={chunk_id}, distance={dist:.4f})"
        )

    print("\nEVIDENCE-BASED SUMMARY:")
    for point in format_evidence_summary(metadatas):
        print(point)

    print("\nSOURCES:")
    for line in format_sources(metadatas):
        print(line)

    print("\n" + "-" * 100)
    print("RAW TOP CHUNK:")
    print(metadatas[0].get("raw_text", "")[:1500])
    print("-" * 100)


def main():
    print("Starting Ask-My-Docs interactive retrieval...")
    print("Type 'exit' to quit.\n")

    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(path="chroma_db", collection_name="httpx_docs")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query:
            print("Please enter a question.")
            continue

        run_query(query, embedder, vector_store)


if __name__ == "__main__":
    main()