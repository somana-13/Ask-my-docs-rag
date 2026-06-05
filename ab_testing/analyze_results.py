from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


RESULTS_PATH = Path("ab_testing/results/ab_test_results.csv")
REPORT_PATH = Path("reports/ab_test_report.md")


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def summarize_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant_name, group in df.groupby("variant_name"):
        in_domain = group[group["question_type"] == "in_domain"]
        out_domain = group[group["question_type"] == "out_of_domain"]

        rows.append(
            {
                "variant_name": variant_name,
                "n_requests": len(group),
                "source_match_rate": safe_rate(in_domain["source_match"]),
                "keyword_match_rate": safe_rate(in_domain["keyword_match"]),
                "out_of_domain_abstention_rate": safe_rate(out_domain["abstained"]),
                "avg_latency_ms": float(group["latency_ms"].mean()),
                "error_rate": safe_rate(group["error"] != ""),
            }
        )

    return pd.DataFrame(rows)


def fisher_test_for_abstention(df: pd.DataFrame):
    """
    Tests whether Variant B abstains on out-of-domain questions
    more often than Variant A.

    This is a tiny demo dataset, so treat the p-value as illustrative,
    not production-grade evidence.
    """
    out_domain = df[df["question_type"] == "out_of_domain"]

    variant_names = sorted(out_domain["variant_name"].unique())

    if len(variant_names) != 2:
        return None

    a_name, b_name = variant_names

    a = out_domain[out_domain["variant_name"] == a_name]
    b = out_domain[out_domain["variant_name"] == b_name]

    table = [
        [int(a["abstained"].sum()), int((1 - a["abstained"]).sum())],
        [int(b["abstained"].sum()), int((1 - b["abstained"]).sum())],
    ]

    odds_ratio, p_value = fisher_exact(table)

    return {
        "variant_a": a_name,
        "variant_b": b_name,
        "contingency_table": table,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def write_report(summary_df: pd.DataFrame, test_result):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    table = summary_df.to_markdown(index=False)

    if test_result:
        stats_section = f"""
## Statistical Test

Because the dataset is small, this test is included as a demonstration of A/B testing methodology rather than a production-grade conclusion.

We used Fisher's exact test to compare out-of-domain abstention between the two variants.

- Variant A: `{test_result["variant_a"]}`
- Variant B: `{test_result["variant_b"]}`
- Contingency table: `{test_result["contingency_table"]}`
- Odds ratio: `{test_result["odds_ratio"]:.4f}`
- p-value: `{test_result["p_value"]:.4f}`
"""
    else:
        stats_section = """
## Statistical Test

Statistical test was skipped because the result file did not contain exactly two variants with out-of-domain examples.
"""

    report = f"""# A/B Test Report

## Objective

This A/B test compares two RAG system variants:

- **Variant A:** Dense retrieval baseline
- **Variant B:** Dense retrieval with evidence-based abstention

The goal is to test whether abstention improves out-of-domain safety while preserving in-domain retrieval quality.

## Variant Summary

{table}

{stats_section}

## Interpretation

The baseline dense retriever returns nearest-neighbor chunks even for unrelated questions. This can create hallucination risk because the system may answer with weak or irrelevant evidence.

The abstention variant adds a confidence threshold based on retrieval distance. If the retrieved evidence is too weak, the system returns no answer.

In this experiment, the abstention variant is expected to improve out-of-domain abstention while maintaining source and keyword match for in-domain documentation questions.

## Interview Summary

I simulated an A/B test comparing a dense retrieval baseline against a dense retrieval system with evidence-based abstention. The experiment measured in-domain source match, keyword match, out-of-domain abstention, latency, and error rate. This showed how an ML system improvement can be evaluated using product-style experimentation rather than only offline accuracy.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"A/B test report written to: {REPORT_PATH}")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "A/B test results not found. Run: python -m ab_testing.ab_test_runner"
        )

    df = pd.read_csv(RESULTS_PATH)
    summary_df = summarize_by_variant(df)
    test_result = fisher_test_for_abstention(df)

    write_report(summary_df, test_result)

    print("\nA/B summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()