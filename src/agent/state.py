from __future__ import annotations

from typing import Any, Literal, TypedDict


RouteType = Literal["simple_search", "decompose_multihop", "clarify"]


class AgentState(TypedDict, total=False):
    """
    Shared state object passed through the LangGraph workflow.

    Each node reads from this state and writes updated fields back into it.
    """

    query: str
    route: RouteType

    sub_questions: list[str]
    subquestion_results: list[dict[str, Any]]

    retrieval_result: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    sources: list[str]

    answer: str
    citations: list[str]

    needs_clarification: bool
    clarification_question: str

    error: str
