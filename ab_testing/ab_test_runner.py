import argparse
import random
import time
from pathlib import Path

import pandas as pd
import yaml

from experiments.run_experiment import (
    build_retriever,
    keywords_match,
    load_chunks,
    load_config,
    normalize_text,
    run_query,
    source_matches,
)


def load_ab_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_variant(split: dict, rng: random.Random) -> str:
    """
    Randomly assigns a request to Variant A or Variant B.
    """
    p_a = split.get("A", 0.5)
    return "A" if rng.random() < p_a else "B"


def initialize_variant(config_path: str):
    """
    Loads one variant config and builds its retriever once.
    """
    config = load_config(config_path)
    chunks = load_chunks(config["data"]["chunks_file"])
    retriever = build_retriever(config, chunks)

    return {
        "config": config,
        "retriever": retriever,
    }


def run_ab_test(config_path: str) -> pd.DataFrame:
    ab_config = load_ab_config(config_path)

    output_dir = Path(ab_config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(ab_config["data"]["eval_file"])

    rng = random.Random(ab_config["traffic"]["random_seed"])
    split = ab_config["traffic"]["split"]

    variants = {
        variant_key: initialize_variant(variant_info["config_path"])
        for variant_key, variant_info in ab_config["variants"].items()
    }

    records = []

    for _, row in eval_df.iterrows():
        question = normalize_text(row["question"])
        expected_source = normalize_text(row.get("expected_source", ""))
        expected_keywords = normalize_text(row.get("expected_keywords", ""))
        question_type = normalize_text(row.get("question_type", "in_domain"))

        assigned_variant = assign_variant(split, rng)
        variant_state = variants[assigned_variant]
        variant_config = variant_state["config"]
        retriever = variant_state["retriever"]

        start = time.time()

        try:
            results = run_query(retriever, question, variant_config)
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
                "variant": assigned_variant,
                "variant_name": ab_config["variants"][assigned_variant]["name"],
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

    output_path = output_dir / "ab_test_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"A/B test results written to: {output_path}")
    print(results_df[["question", "variant_name", "abstained", "source_match", "keyword_match"]])

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="ab_testing/ab_test_config.yaml",
        help="Path to A/B test config YAML file.",
    )
    args = parser.parse_args()

    run_ab_test(args.config)


if __name__ == "__main__":
    main()