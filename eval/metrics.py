from __future__ import annotations

from typing import Any


def normalize_source(source: str) -> str:
    """
    Normalize source strings so we can compare expected files like
    'authentication.md' against retrieved source lines like
    '- authentication.md | NetRC authentication'.
    """
    return source.lower().replace("-", "").strip()


def source_line_contains_expected(source_line: str, expected_source: str) -> bool:
    return normalize_source(expected_source) in normalize_source(source_line)


def compute_router_correct(result: dict[str, Any], example: dict[str, Any]) -> bool:
    return result.get("route") == example.get("expected_route")


def compute_retrieval_precision_at_k(
    result: dict[str, Any],
    example: dict[str, Any],
    k: int = 5,
) -> float:
    """
    Measures how many expected source files appeared in the top-k retrieved sources.

    Example:
    expected_sources = ['authentication.md', 'timeouts.md']
    retrieved sources include both -> 1.0
    retrieved sources include one -> 0.5
    """
    expected_sources = example.get("expected_sources", [])

    if not expected_sources:
        return 1.0

    retrieved_sources = result.get("sources", [])[:k]

    matched = 0
    for expected in expected_sources:
        if any(source_line_contains_expected(line, expected) for line in retrieved_sources):
            matched += 1

    return matched / len(expected_sources)


def compute_citation_coverage(
    result: dict[str, Any],
    example: dict[str, Any],
) -> float:
    """
    Measures whether generated citations cover expected sources.
    """
    expected_sources = example.get("expected_sources", [])

    if not expected_sources:
        return 1.0

    citations = result.get("citations", [])

    matched = 0
    for expected in expected_sources:
        if any(source_line_contains_expected(citation, expected) for citation in citations):
            matched += 1

    return matched / len(expected_sources)


def compute_abstention_correct(result: dict[str, Any], example: dict[str, Any]) -> bool:
    """
    For now, answerable=false examples are expected to route to clarify.
    """
    answerable = example.get("answerable", True)

    if answerable:
        return result.get("route") != "clarify"

    return result.get("route") == "clarify"
