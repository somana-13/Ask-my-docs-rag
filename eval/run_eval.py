from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from src.agent.graph import build_graph

from eval.metrics import (
    compute_abstention_correct,
    compute_citation_coverage,
    compute_retrieval_precision_at_k,
    compute_router_correct,
)
from eval.nli_faithfulness import evaluate_faithfulness


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def run_eval(dataset_path: str, output_path: str) -> dict[str, Any]:
    examples = load_dataset(dataset_path)
    app = build_graph()

    detailed_results = []

    for example in examples:
        question = example["question"]

        start_time = time.perf_counter()
        result = app.invoke({"query": question})
        latency_ms = (time.perf_counter() - start_time) * 1000

        answer = result.get("answer", "")
        evidence_chunks = result.get("retrieved_chunks", [])

        # Only score faithfulness when the system attempted an evidence-based answer.
        if result.get("route") == "clarify" or not evidence_chunks:
            faithfulness = {
                "faithfulness_score": 1.0 if not example.get("answerable", True) else 0.0,
                "num_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "contradicted_claims": 0,
                "claim_results": [],
            }
        else:
            faithfulness = evaluate_faithfulness(
                answer=answer,
                evidence_chunks=evidence_chunks,
            )

        router_correct = compute_router_correct(result, example)
        retrieval_precision_at_5 = compute_retrieval_precision_at_k(result, example, k=5)
        citation_coverage = compute_citation_coverage(result, example)
        abstention_correct = compute_abstention_correct(result, example)

        detailed_results.append(
            {
                "id": example.get("id"),
                "question": question,
                "question_type": example.get("question_type"),
                "expected_route": example.get("expected_route"),
                "actual_route": result.get("route"),
                "answerable": example.get("answerable"),
                "expected_sources": example.get("expected_sources", []),
                "retrieved_sources": result.get("sources", []),
                "citations": result.get("citations", []),
                "sub_questions": result.get("sub_questions", []),
                "answer": answer,
                "latency_ms": latency_ms,
                "router_correct": router_correct,
                "retrieval_precision_at_5": retrieval_precision_at_5,
                "citation_coverage": citation_coverage,
                "abstention_correct": abstention_correct,
                "faithfulness_score": faithfulness["faithfulness_score"],
                "faithfulness": faithfulness,
            }
        )

    summary = {
        "num_examples": len(detailed_results),
        "router_accuracy": safe_mean([float(x["router_correct"]) for x in detailed_results]),
        "retrieval_precision_at_5": safe_mean(
            [float(x["retrieval_precision_at_5"]) for x in detailed_results]
        ),
        "citation_coverage": safe_mean([float(x["citation_coverage"]) for x in detailed_results]),
        "abstention_accuracy": safe_mean(
            [float(x["abstention_correct"]) for x in detailed_results]
        ),
        "faithfulness_score": safe_mean([float(x["faithfulness_score"]) for x in detailed_results]),
        "avg_latency_ms": safe_mean([float(x["latency_ms"]) for x in detailed_results]),
    }

    output = {
        "summary": summary,
        "results": detailed_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation.")
    parser.add_argument(
        "--dataset",
        default="eval/golden_dataset.json",
        help="Path to golden eval dataset.",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_results.json",
        help="Path to save detailed eval results.",
    )
    args = parser.parse_args()

    output = run_eval(dataset_path=args.dataset, output_path=args.output)
    summary = output["summary"]

    print("\n" + "=" * 100)
    print("AGENTIC RAG EVALUATION SUMMARY")
    print("=" * 100)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
