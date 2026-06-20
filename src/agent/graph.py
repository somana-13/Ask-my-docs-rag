from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    answer_node,
    clarify_node,
    decompose_node,
    retrieve_subquestions_node,
    router_node,
    simple_search_node,
    synthesis_node,
)
from src.agent.state import AgentState


def route_from_state(state: AgentState) -> str:
    """
    Choose the next graph node based on the router decision.
    """
    return state.get("route", "simple_search")


def build_graph():
    """
    Build the LangGraph workflow.

    Current Day 3 graph:

    router
      ├── simple_search -> answer -> END
      ├── decompose_multihop -> decompose -> retrieve_subquestions -> synthesis -> END
      └── clarify -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("simple_search", simple_search_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve_subquestions", retrieve_subquestions_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_from_state,
        {
            "simple_search": "simple_search",
            "decompose_multihop": "decompose",
            "clarify": "clarify",
        },
    )

    graph.add_edge("simple_search", "answer")
    graph.add_edge("answer", END)

    graph.add_edge("decompose", "retrieve_subquestions")
    graph.add_edge("retrieve_subquestions", "synthesis")
    graph.add_edge("synthesis", END)

    graph.add_edge("clarify", END)

    return graph.compile()
