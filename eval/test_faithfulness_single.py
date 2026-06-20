from __future__ import annotations

import argparse
import json

from src.agent.graph import build_graph
from eval.nli_faithfulness import evaluate_faithfulness


def main() -> None:
    parser = argparse.ArgumentParser(description="Test NLI faithfulness on one query.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    app = build_graph()
    result = app.invoke({"query": args.query})

    answer = result.get("answer", "")
    evidence_chunks = result.get("retrieved_chunks", [])

    faithfulness = evaluate_faithfulness(
        answer=answer,
        evidence_chunks=evidence_chunks,
    )

    output = {
        "query": result.get("query"),
        "route": result.get("route"),
        "sub_questions": result.get("sub_questions", []),
        "sources": result.get("sources", []),
        "answer": answer,
        "faithfulness": faithfulness,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
