from __future__ import annotations

import argparse
import sys

from eval.run_eval import run_eval


DEFAULT_THRESHOLDS = {
    "router_accuracy": 0.90,
    "retrieval_precision_at_5": 0.90,
    "citation_coverage": 0.90,
    "abstention_accuracy": 0.90,
    "faithfulness_score": 0.80,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CI regression gate for Agentic RAG eval.")
    parser.add_argument(
        "--dataset",
        default="eval/ci_smoke_set.json",
        help="Path to CI smoke evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        default="reports/ci_eval_results.json",
        help="Path to save CI eval results.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["faithfulness_score"],
    )
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds["faithfulness_score"] = args.faithfulness_threshold

    output = run_eval(dataset_path=args.dataset, output_path=args.output)
    summary = output["summary"]

    print("\n" + "=" * 100)
    print("CI EVALUATION GATE")
    print("=" * 100)

    failed = []

    for metric, threshold in thresholds.items():
        value = float(summary.get(metric, 0.0))
        status = "PASS" if value >= threshold else "FAIL"
        print(f"{metric}: {value:.4f} | threshold: {threshold:.4f} | {status}")

        if value < threshold:
            failed.append((metric, value, threshold))

    print(f"\nSaved CI eval results to: {args.output}")

    if failed:
        print("\nCI gate failed:")
        for metric, value, threshold in failed:
            print(f"- {metric}: {value:.4f} < {threshold:.4f}")
        sys.exit(1)

    print("\nCI gate passed.")


if __name__ == "__main__":
    main()
