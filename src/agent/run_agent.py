from __future__ import annotations

import argparse

from src.agent.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic RAG workflow.")
    parser.add_argument("--query", required=True, help="User question to answer.")
    args = parser.parse_args()

    app = build_graph()

    result = app.invoke(
        {
            "query": args.query,
        }
    )

    print("\n" + "=" * 100)
    print("AGENTIC RAG RESULT")
    print("=" * 100)

    print(f"\nQuery:\n{result.get('query')}")

    print(f"\nRoute:\n{result.get('route')}")

    if result.get("sub_questions"):
     print("\nSub-questions:")
     for sub_question in result.get("sub_questions", []):
        print(f"- {sub_question}")

    print("\nSources:")
    for source in result.get("sources", []):
        print(source)

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nCitations:")
    for citation in result.get("citations", []):
        print(f"- {citation}")


if __name__ == "__main__":
    main()
