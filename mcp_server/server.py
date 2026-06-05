from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from experiments.run_experiment import (
    build_retriever,
    get_source_from_result,
    get_text_from_result,
    load_chunks,
    load_config,
    run_query,
)


mcp = FastMCP("ask-my-docs-rag")

DEFAULT_VARIANT = "dense_abstention"

VARIANT_CONFIGS = {
    "dense_baseline": "experiments/configs/dense_baseline.yaml",
    "dense_abstention": "experiments/configs/dense_abstention.yaml",
    "hybrid_rerank": "experiments/configs/hybrid_rerank.yaml",
}


_VARIANT_CACHE: dict[str, dict[str, Any]] = {}


def load_variant(variant: str) -> dict[str, Any]:
    """
    Loads and caches a RAG variant so repeated MCP calls do not rebuild retrievers.
    """
    if variant not in VARIANT_CONFIGS:
        valid = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(f"Unknown variant '{variant}'. Valid variants: {valid}")

    if variant in _VARIANT_CACHE:
        return _VARIANT_CACHE[variant]

    config = load_config(VARIANT_CONFIGS[variant])
    chunks = load_chunks(config["data"]["chunks_file"])
    retriever = build_retriever(config, chunks)

    state = {
        "config": config,
        "chunks": chunks,
        "retriever": retriever,
    }

    _VARIANT_CACHE[variant] = state
    return state


def format_results(results: list[dict], max_chars: int = 700) -> list[dict]:
    """
    Converts retrieved chunks into clean JSON-serializable outputs.
    """
    formatted = []

    for i, result in enumerate(results, 1):
        text = get_text_from_result(result)
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}

        source = (
            metadata.get("source_name")
            or metadata.get("source")
            or get_source_from_result(result)
        )

        section = (
            metadata.get("section_title")
            or metadata.get("section")
            or ""
        )

        formatted.append(
            {
                "rank": i,
                "source": source,
                "section": section,
                "text_preview": text[:max_chars],
            }
        )

    return formatted


@mcp.tool()
def list_variants() -> dict:
    """
    List available RAG pipeline variants.
    """
    return {
        "default_variant": DEFAULT_VARIANT,
        "available_variants": sorted(VARIANT_CONFIGS.keys()),
    }


@mcp.tool()
def list_sources() -> dict:
    """
    List available documentation sources in the indexed corpus.
    """
    state = load_variant(DEFAULT_VARIANT)
    chunks = state["chunks"]

    sources = sorted(
        {
            chunk.get("source_name")
            or chunk.get("source")
            or chunk.get("source_file")
            or chunk.get("file")
            or chunk.get("filename")
            or chunk.get("source_path")
            or chunk.get("path")
            or ""
            for chunk in chunks
        }
    )

    sources = [source for source in sources if source]

    return {
        "num_sources": len(sources),
        "sources": sources,
    }


@mcp.tool()
def search_docs(question: str, variant: str = DEFAULT_VARIANT, top_k: int = 5) -> dict:
    """
    Search the indexed documentation and return grounded evidence chunks.
    """
    state = load_variant(variant)
    config = dict(state["config"])
    config["retriever"] = dict(config["retriever"])
    config["retriever"]["top_k"] = top_k

    results = run_query(state["retriever"], question, config)

    return {
        "question": question,
        "variant": variant,
        "num_results": len(results),
        "results": format_results(results),
    }


@mcp.tool()
def answer_question(question: str, variant: str = DEFAULT_VARIANT) -> dict:
    """
    Produce a simple grounded answer using retrieved documentation evidence.

    This is intentionally extractive/simple for now. It does not call an LLM.
    """
    search_output = search_docs(question=question, variant=variant, top_k=5)
    results = search_output["results"]

    if not results:
        return {
            "question": question,
            "variant": variant,
            "answered": False,
            "answer": "I could not find strong enough supporting evidence in the indexed documentation.",
            "citations": [],
        }

    citations = [
        {
            "source": result["source"],
            "section": result["section"],
        }
        for result in results[:3]
    ]

    evidence_text = "\n\n".join(
        f"[{i}] Source: {result['source']} | Section: {result['section']}\n"
        f"{result['text_preview']}"
        for i, result in enumerate(results[:3], 1)
    )

    answer = (
        "I found relevant documentation evidence. "
        "Use the cited sources below to answer the question:\n\n"
        f"{evidence_text}"
    )

    return {
        "question": question,
        "variant": variant,
        "answered": True,
        "answer": answer,
        "citations": citations,
    }


@mcp.tool()
def get_experiment_summary() -> dict:
    """
    Return the generated MLflow experiment summary report.
    """
    report_path = Path("reports/mlflow_experiment_summary.md")

    if not report_path.exists():
        return {
            "available": False,
            "message": "Experiment summary report not found. Run python -m experiments.compare_runs first.",
        }

    return {
        "available": True,
        "report": report_path.read_text(encoding="utf-8"),
    }


if __name__ == "__main__":
    mcp.run()