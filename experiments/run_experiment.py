import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LocalReranker
from src.retrieval.vector_store import ChromaVectorStore


_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_file: str) -> list[dict]:
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """
    Converts processed chunk dictionaries into LangChain Document objects
    so LangChain BM25Retriever can index them.
    """
    documents = []

    for chunk in chunks:
        text = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("document")
            or chunk.get("page_content")
            or ""
        )

        source = (
            chunk.get("source")
            or chunk.get("source_file")
            or chunk.get("file")
            or chunk.get("filename")
            or chunk.get("path")
            or ""
        )

        section = chunk.get("section") or chunk.get("heading") or ""
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or ""

        metadata = {
            "source": source,
            "section": section,
            "chunk_id": chunk_id,
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_retrieval_results(results):
    """
    Converts different retriever outputs into a common list[dict] format.

    Chroma returns:
    {
        "ids": [[...]],
        "documents": [[...]],
        "metadatas": [[...]],
        "distances": [[...]]
    }

    We convert it into:
    [
        {
            "id": "...",
            "text": "...",
            "metadata": {...},
            "distance": ...
        }
    ]
    """
    if isinstance(results, list):
        normalized = []

        for result in results:
            if isinstance(result, Document):
                metadata = dict(result.metadata or {})
                metadata.setdefault("raw_text", result.page_content)
                metadata.setdefault("source_name", metadata.get("source", ""))
                metadata.setdefault("section_title", metadata.get("section", ""))

                normalized.append(
                    {
                        "id": metadata.get("chunk_id", ""),
                        "text": result.page_content,
                        "metadata": metadata,
                        "distance": None,
                    }
                )
            else:
                normalized.append(result)

        return normalized

    if isinstance(results, dict) and "documents" in results:
        docs = results.get("documents") or [[]]
        ids = results.get("ids") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        docs = docs[0] if docs and isinstance(docs[0], list) else docs
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        metadatas = (
            metadatas[0]
            if metadatas and isinstance(metadatas[0], list)
            else metadatas
        )
        distances = (
            distances[0]
            if distances and isinstance(distances[0], list)
            else distances
        )

        normalized = []

        for i, doc in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            metadata = dict(metadata or {})

            if not metadata.get("source"):
                for line in str(doc).splitlines():
                    line = line.strip()
                    if line.lower().startswith("source:"):
                        metadata["source"] = line.split(":", 1)[1].strip()
                        break

            if not metadata.get("section"):
                for line in str(doc).splitlines():
                    line = line.strip()
                    if line.lower().startswith("section:"):
                        metadata["section"] = line.split(":", 1)[1].strip()
                        break

            metadata.setdefault("raw_text", doc)
            metadata.setdefault("source_name", metadata.get("source", ""))
            metadata.setdefault("section_title", metadata.get("section", ""))

            normalized.append(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "text": doc,
                    "metadata": metadata,
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return normalized

    return []


def get_source_from_result(result) -> str:
    """
    Extracts source filename from metadata or from text like:
    Source: authentication.md
    """
    if not isinstance(result, dict):
        return ""

    metadata = result.get("metadata", {})

    if isinstance(metadata, dict):
        source = (
            metadata.get("source")
            or metadata.get("file")
            or metadata.get("filename")
            or metadata.get("source_file")
            or metadata.get("path")
            or ""
        )

        if source:
            return source

    text = get_text_from_result(result)

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("source:"):
            return line.split(":", 1)[1].strip()

    return result.get("source", "")


def get_text_from_result(result) -> str:
    """
    Extracts document text from normalized retriever result.
    """
    if isinstance(result, dict):
        return (
            result.get("text")
            or result.get("content")
            or result.get("document")
            or result.get("page_content")
            or ""
        )

    return str(result)


# ----------------- Abstention & Distance helpers -----------------

def get_distance_from_result(result):
    """
    Extracts vector distance when available.
    Lower distance usually means stronger semantic match in Chroma.
    """
    if isinstance(result, dict):
        return result.get("distance")
    return None


def should_abstain(results: list, config: dict) -> bool:
    """
    Decides whether to abstain based on retrieval confidence.

    For Chroma-style distances:
    - lower distance = better match
    - higher distance = weaker evidence

    If the best result distance is above the threshold, we abstain.
    """
    abstention_config = config.get("abstention", {})

    if not abstention_config.get("enabled", False):
        return False

    if not results:
        return True

    threshold = abstention_config.get("threshold")

    if threshold is None:
        return False

    distances = [
        get_distance_from_result(result)
        for result in results
        if get_distance_from_result(result) is not None
    ]

    if not distances:
        return False

    best_distance = min(distances)

    return best_distance > threshold


def source_matches(results: list, expected_source: str) -> bool:
    if not expected_source:
        return False

    expected_source = expected_source.lower()

    for result in results:
        source = get_source_from_result(result).lower()
        if expected_source in source:
            return True

    return False


def keywords_match(results: list, expected_keywords: str) -> bool:
    if not expected_keywords:
        return False

    keywords = [
        keyword.strip().lower()
        for keyword in expected_keywords.split(",")
        if keyword.strip()
    ]

    evidence_text = " ".join(get_text_from_result(result) for result in results).lower()

    return any(keyword in evidence_text for keyword in keywords)


def build_retriever(config: dict, chunks: list[dict]):
    retriever_type = config["retriever"]["type"]
    top_k = config["retriever"]["top_k"]

    if retriever_type == "dense":
        return ChromaVectorStore()

    if retriever_type == "bm25":
        documents = chunks_to_documents(chunks)

        if hasattr(BM25Retriever, "from_documents"):
            retriever = BM25Retriever.from_documents(documents)
        else:
            retriever = BM25Retriever(docs=documents)

        if hasattr(retriever, "k"):
            retriever.k = top_k

        return retriever

    if retriever_type == "hybrid":
        dense_retriever = ChromaVectorStore()
        documents = chunks_to_documents(chunks)

        if hasattr(BM25Retriever, "from_documents"):
            bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            bm25_retriever = BM25Retriever(docs=documents)

        if hasattr(bm25_retriever, "k"):
            bm25_retriever.k = top_k

        try:
            return HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
            )
        except TypeError:
            try:
                return HybridRetriever(dense_retriever, bm25_retriever)
            except TypeError:
                return {
                    "dense_retriever": dense_retriever,
                    "bm25_retriever": bm25_retriever,
                }

    raise ValueError(f"Unsupported retriever type: {retriever_type}")


def run_query(retriever, question: str, config: dict):
    top_k = config["retriever"]["top_k"]
    retriever_type = config["retriever"]["type"]

    if retriever_type == "dense":
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(question).tolist()

        if hasattr(retriever, "query"):
            try:
                results = retriever.query(query_embedding, top_k=top_k)
            except TypeError:
                results = retriever.query(query_embedding)
        elif hasattr(retriever, "search"):
            try:
                results = retriever.search(query_embedding, top_k=top_k)
            except TypeError:
                results = retriever.search(query_embedding)
        elif hasattr(retriever, "retrieve"):
            try:
                results = retriever.retrieve(query_embedding, top_k=top_k)
            except TypeError:
                results = retriever.retrieve(query_embedding)
        else:
            raise AttributeError("Dense retriever must have query(), search(), or retrieve().")
    else:
        if hasattr(retriever, "retrieve"):
            try:
                results = retriever.retrieve(question, top_k=top_k)
            except TypeError:
                results = retriever.retrieve(question)
        elif hasattr(retriever, "search"):
            try:
                results = retriever.search(question, top_k=top_k)
            except TypeError:
                results = retriever.search(question)
        elif hasattr(retriever, "query"):
            try:
                results = retriever.query(question, top_k=top_k)
            except TypeError:
                results = retriever.query(question)
        elif hasattr(retriever, "invoke"):
            results = retriever.invoke(question)
        elif isinstance(retriever, dict) and "dense_retriever" in retriever and "bm25_retriever" in retriever:
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.encode(question).tolist()

            try:
                dense_results = retriever["dense_retriever"].query(query_embedding, top_k=top_k)
            except TypeError:
                dense_results = retriever["dense_retriever"].query(query_embedding)
            bm25_results = retriever["bm25_retriever"].invoke(question)

            dense_results = normalize_retrieval_results(dense_results)
            bm25_results = normalize_retrieval_results(bm25_results)

            seen = set()
            results = []
            for result in dense_results + bm25_results:
                result_id = result.get("id") or get_text_from_result(result)[:120]
                if result_id not in seen:
                    seen.add(result_id)
                    results.append(result)

            results = results[:top_k]
        else:
            raise AttributeError("Retriever must have retrieve(), search(), query(), or invoke().")

    results = normalize_retrieval_results(results)

    if config["reranker"]["enabled"]:
        reranker = LocalReranker(
            model_name=config["reranker"].get(
                "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )

        if hasattr(reranker, "rerank"):
            try:
                results = reranker.rerank(question, results, top_k=top_k)
            except TypeError:
                results = reranker.rerank(question, results)
        else:
            raise AttributeError("Reranker must have rerank().")

        results = normalize_retrieval_results(results)

    if should_abstain(results, config):
        return []

    return results


def evaluate(config: dict) -> tuple[pd.DataFrame, dict]:
    eval_df = pd.read_csv(config["data"]["eval_file"])
    chunks = load_chunks(config["data"]["chunks_file"])

    retriever = build_retriever(config, chunks)

    records = []

    for _, row in eval_df.iterrows():
        question = normalize_text(row["question"])
        expected_source = normalize_text(row.get("expected_source", ""))
        expected_keywords = normalize_text(row.get("expected_keywords", ""))
        question_type = normalize_text(row.get("question_type", "in_domain"))

        start = time.time()
        try:
            results = run_query(retriever, question, config)
            latency_ms = (time.time() - start) * 1000
            error = ""
        except Exception as e:
            results = []
            latency_ms = (time.time() - start) * 1000
            error = str(e)

        has_source_match = source_matches(results, expected_source)
        has_keyword_match = keywords_match(results, expected_keywords)

        abstained = len(results) == 0

        records.append(
            {
                "question": question,
                "question_type": question_type,
                "expected_source": expected_source,
                "expected_keywords": expected_keywords,
                "source_match": int(has_source_match),
                "keyword_match": int(has_keyword_match),
                "abstained": int(abstained),
                "latency_ms": latency_ms,
                "num_results": len(results),
                "error": error,
            }
        )

    results_df = pd.DataFrame(records)

    in_domain = results_df[results_df["question_type"] == "in_domain"]
    out_domain = results_df[results_df["question_type"] == "out_of_domain"]

    metrics = {
        "source_match_rate": float(in_domain["source_match"].mean())
        if len(in_domain)
        else 0.0,
        "keyword_match_rate": float(in_domain["keyword_match"].mean())
        if len(in_domain)
        else 0.0,
        "avg_latency_ms": float(results_df["latency_ms"].mean()),
        "avg_num_results": float(results_df["num_results"].mean()),
        "out_of_domain_abstention_rate": float(out_domain["abstained"].mean())
        if len(out_domain)
        else 0.0,
        "error_rate": float((results_df["error"] != "").mean()),
    }

    return results_df, metrics


def log_to_mlflow(config: dict, config_path: str, results_df: pd.DataFrame, metrics: dict):
    mlflow.set_experiment(config["experiment_name"])

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    run_name = config["run_name"]
    predictions_path = results_dir / f"{run_name}_predictions.csv"
    summary_path = results_dir / f"{run_name}_summary.json"

    results_df.to_csv(predictions_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("retriever_type", config["retriever"]["type"])
        mlflow.log_param("top_k", config["retriever"]["top_k"])
        mlflow.log_param("reranker_enabled", config["reranker"]["enabled"])
        mlflow.log_param("abstention_enabled", config["abstention"]["enabled"])
        mlflow.log_param("abstention_threshold", config["abstention"]["threshold"])

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(config_path)
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(summary_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results_df, metrics = evaluate(config)
    log_to_mlflow(config, args.config, results_df, metrics)

    print("\nExperiment complete.")
    print(f"Run name: {config['run_name']}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()