# Ask My Docs: Agentic RAG Evaluation Platform

A production-style **Agentic Retrieval-Augmented Generation (RAG)** system over technical documentation. The project started as a grounded documentation QA system and was upgraded into a LangGraph-based agent with routing, multi-hop decomposition, clarification/abstention, local NLI faithfulness evaluation, MLflow tracking, A/B testing, MCP tools, and CI regression gates.

The current corpus indexes a curated subset of the **HTTPX documentation** and answers only when it can retrieve grounded evidence from the indexed docs.

---

## Project Highlights

- Built a document QA RAG pipeline over HTTPX docs using Sentence Transformers and ChromaDB.
- Added BM25, hybrid retrieval, cross-encoder reranking, and evidence-based abstention.
- Upgraded the fixed RAG pipeline into a **LangGraph agent** with conditional routing.
- Added a **multi-hop decomposition path** for questions requiring evidence from multiple documentation sections.
- Built a reproducible evaluation harness with router accuracy, retrieval precision@5, citation coverage, abstention accuracy, latency, and local NLI faithfulness.
- Improved local NLI faithfulness from **0.7367 → 0.9500** after failure analysis and claim-extraction fixes.
- Added a GitHub Actions CI gate that fails when quality metrics regress below thresholds.
- Exposed the RAG pipeline through MCP tools for agent-compatible integration.

---

## Architecture

```text
User query
   ↓
LangGraph router
   ├── simple_search
   │      ↓
   │   retrieval tool
   │      ↓
   │   evidence-based answer
   │
   ├── decompose_multihop
   │      ↓
   │   decompose into sub-questions
   │      ↓
   │   retrieve evidence per sub-question
   │      ↓
   │   synthesize final answer
   │
   └── clarify
          ↓
       ask for clarification / abstain

Final answer
   ↓
claim extraction
   ↓
local NLI cross-encoder faithfulness scoring
   ↓
eval report + CI regression gate
```

---

## Corpus

Indexed HTTPX documentation files include:

- `authentication.md`
- `quickstart.md`
- `timeouts.md`
- `ssl.md`
- `clients.md`
- `async.md`
- `proxies.md`
- `transports.md`
- `exceptions.md`
- `environment_variables.md`
- `index.md`

---

## Tech Stack

- Python
- LangGraph
- Sentence Transformers
- ChromaDB
- BM25 / `rank-bm25`
- Cross-encoder reranking
- Local NLI cross-encoder faithfulness scoring
- MLflow
- GitHub Actions
- MCP
- Pytest
- Pandas / SciPy

---

## Features

### Phase 1: RAG Baseline

- Ingest markdown documentation.
- Split documents into section-aware chunks.
- Preserve metadata such as source file, section title, and chunk id.
- Build a persistent ChromaDB vector index.
- Run dense semantic retrieval with source-aware evidence snippets.

### Phase 2: Better Retrieval

- Add BM25 lexical retrieval.
- Combine dense and BM25 results with hybrid retrieval.
- Rerank candidates using a cross-encoder.
- Filter weak evidence.
- Abstain when retrieved evidence is not strong enough.

### Phase 3: MLflow Experiment Tracking

- Define config-driven retrieval experiments.
- Track retriever type, top-k, reranker settings, abstention settings, latency, and error rate.
- Save prediction CSVs and summary JSON artifacts.
- Generate comparison reports across retrieval variants.

### Phase 4: A/B Testing Simulation

- Compare a dense retrieval baseline against an abstention-enabled treatment.
- Measure source match, keyword match, out-of-domain abstention, latency, and error rate.
- Generate an offline A/B test report.

### Phase 5: MCP Tool Server

- Expose RAG functionality through MCP tools.
- Support agent-ready documentation search and grounded answering.
- Preserve out-of-domain refusal behavior through the default `dense_abstention` variant.

### Phase 6: Agentic RAG with LangGraph

- Add a LangGraph router node for conditional workflow selection.
- Route queries into `simple_search`, `decompose_multihop`, or `clarify` paths.
- Decompose multi-hop questions into retrieval-friendly sub-questions.
- Retrieve evidence separately for each sub-question.
- Synthesize answers from multiple evidence sources.
- Add deterministic clarification/out-of-corpus handling for vague or unsupported questions.

### Phase 7: Faithfulness Evaluation + CI Regression Gate

- Build a golden evaluation set covering simple, multi-hop, comparison, clarification, and unanswerable questions.
- Extract factual claims from generated answers.
- Score claims against retrieved evidence using a local NLI cross-encoder.
- Track faithfulness, router accuracy, retrieval precision@5, citation coverage, abstention accuracy, and latency.
- Add a GitHub Actions CI gate that fails when metrics fall below thresholds.

---

## Project Structure

```text
ask-my-docs-rag/
├── .github/
│   └── workflows/
│       └── eval.yml
├── ab_testing/
│   ├── ab_test_config.yaml
│   ├── ab_test_runner.py
│   ├── analyze_results.py
│   └── results/
├── chroma_db/
├── data/
│   ├── raw/httpx/
│   ├── processed/
│   │   ├── ingested_docs.json
│   │   └── chunks.json
│   └── eval/
│       └── eval_questions.csv
├── eval/
│   ├── golden_dataset.json
│   ├── ci_smoke_set.json
│   ├── claim_extractor.py
│   ├── nli_faithfulness.py
│   ├── metrics.py
│   ├── run_eval.py
│   ├── ci_gate.py
│   └── test_faithfulness_single.py
├── experiments/
│   ├── configs/
│   │   ├── dense_baseline.yaml
│   │   ├── dense_abstention.yaml
│   │   └── hybrid_rerank.yaml
│   ├── results/
│   ├── run_experiment.py
│   └── compare_runs.py
├── mcp_server/
│   ├── __init__.py
│   └── server.py
├── reports/
│   ├── mlflow_experiment_summary.md
│   ├── ab_test_report.md
│   └── eval_results.md
├── scripts/
│   ├── ingest_docs.py
│   ├── build_chunks.py
│   ├── build_index.py
│   ├── query_index.py
│   ├── query_bm25.py
│   ├── query_hybrid.py
│   └── query_reranked.py
├── src/
│   ├── agent/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── run_agent.py
│   ├── chunking/
│   ├── embeddings/
│   ├── ingestion/
│   ├── retrieval/
│   └── tools/
│       └── retrieval_tool.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Setup

```bash
cd ask-my-docs-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install langgraph langchain-core sentence-transformers scikit-learn torch
```

---

## Build the RAG Index

```bash
python -m scripts.ingest_docs
python -m scripts.build_chunks
python -m scripts.build_index
```

The repository includes a small ChromaDB index so the agentic eval and CI gate can run reproducibly.

---

## Run Manual Retrieval

```bash
python -m scripts.query_index
python -m scripts.query_bm25
python -m scripts.query_hybrid
python -m scripts.query_reranked
```

Example in-domain queries:

- How do I configure authentication in HTTPX?
- What authentication methods does HTTPX support?
- How do timeouts work in HTTPX?
- How do I configure SSL certificates?

Example out-of-domain query:

- How do I fine-tune a BERT model?

Expected abstention response:

```text
I could not find strong enough supporting evidence in the indexed documentation.
```

---

## Run the LangGraph Agent

Run a simple query:

```bash
python -m src.agent.run_agent --query "How do I configure authentication in HTTPX?"
```

Run a multi-hop query:

```bash
python -m src.agent.run_agent --query "How do I configure authentication and timeouts in HTTPX?"
```

Expected route:

```text
decompose_multihop
```

Run a vague query:

```bash
python -m src.agent.run_agent --query "Explain"
```

Expected route:

```text
clarify
```

---

## Run Faithfulness Evaluation

Run a single-query faithfulness test:

```bash
python -m eval.test_faithfulness_single --query "How do I configure authentication and timeouts in HTTPX?"
```

Run the full golden evaluation set:

```bash
python -m eval.run_eval
```

Detailed results are saved to:

```text
reports/eval_results.json
```

---

## Agentic RAG Evaluation Results

Final evaluation on the initial 10-question golden set:

| Metric | Score |
|---|---:|
| Router accuracy | 1.0000 |
| Retrieval precision@5 | 1.0000 |
| Citation coverage | 1.0000 |
| Abstention accuracy | 1.0000 |
| Faithfulness score | 0.9500 |
| Average latency | 175.57 ms |

Improvement after failure analysis:

| Metric | Before | After |
|---|---:|---:|
| Router accuracy | 0.8000 | 1.0000 |
| Retrieval precision@5 | 0.9500 | 1.0000 |
| Citation coverage | 0.8833 | 1.0000 |
| Abstention accuracy | 0.8000 | 1.0000 |
| Faithfulness score | 0.7367 | 0.9500 |
| Average latency | 270.91 ms | 175.57 ms |

Key fixes:

- Added punctuation-normalized routing for vague clarification queries.
- Added out-of-corpus detection for cloud/deployment questions.
- Cleaned retrieved markdown before answer generation.
- Improved claim extraction to remove answer-template prefixes before NLI scoring.
- Removed incomplete markdown/list fragments from faithfulness evaluation.

---

## Run the CI Regression Gate Locally

```bash
python -m eval.ci_gate
```

The CI smoke set checks:

| Metric | Threshold |
|---|---:|
| Router accuracy | 0.90 |
| Retrieval precision@5 | 0.90 |
| Citation coverage | 0.90 |
| Abstention accuracy | 0.90 |
| Faithfulness score | 0.80 |

The gate exits with code `1` if any metric falls below threshold.

---

## GitHub Actions

The workflow is defined in:

```text
.github/workflows/eval.yml
```

It runs on pushes and pull requests to `main` and performs:

1. dependency installation
2. syntax checks
3. Agentic RAG CI smoke evaluation
4. metric threshold validation

---

## Run MLflow Experiments

Run individual experiment variants:

```bash
python -m experiments.run_experiment --config experiments/configs/dense_baseline.yaml
python -m experiments.run_experiment --config experiments/configs/dense_abstention.yaml
python -m experiments.run_experiment --config experiments/configs/hybrid_rerank.yaml
```

Start the MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Generate the experiment summary report:

```bash
python -m experiments.compare_runs
```

Report output:

```text
reports/mlflow_experiment_summary.md
```

---

## Current MLflow Results

| Run | Source Match | Keyword Match | OOD Abstention | Avg Latency | Error Rate |
|---|---:|---:|---:|---:|---:|
| `dense_baseline` | 1.0000 | 1.0000 | 0.0000 | ~244.7 ms | 0.0000 |
| `dense_abstention` | 1.0000 | 1.0000 | 1.0000 | ~222.7 ms | 0.0000 |
| `hybrid_rerank` | 1.0000 | 1.0000 | 1.0000 | ~338.0 ms | 0.0000 |

Best current variant:

```text
dense_abstention
```

Why: it preserves in-domain retrieval quality, improves out-of-domain abstention, and has lower latency than the hybrid reranked pipeline on the current evaluation set.

---

## Run A/B Testing Simulation

Run the offline A/B test:

```bash
python -m ab_testing.ab_test_runner
```

Analyze results:

```bash
python -m ab_testing.analyze_results
```

Report output:

```text
reports/ab_test_report.md
```

A/B setup:

| Variant | System |
|---|---|
| A | Dense retrieval baseline |
| B | Dense retrieval with evidence-based abstention |

Primary metric:

- out-of-domain abstention rate

Guardrail metrics:

- source match rate
- keyword match rate
- latency
- error rate

---

## Run MCP Server

The project exposes the RAG pipeline through an MCP server.

Available MCP tools:

- `list_variants()`
- `list_sources()`
- `search_docs(question, variant, top_k)`
- `answer_question(question, variant)`
- `get_experiment_summary()`

Start the MCP server:

```bash
python -m mcp_server.server
```

The server waits for an MCP client over stdio. This is expected behavior.

Test MCP tools directly:

```bash
python - <<'PY'
from mcp_server.server import list_variants, list_sources, search_docs, answer_question

print(list_variants())
print(list_sources())
print(search_docs("How do I configure authentication in HTTPX?", "dense_abstention", 3))
print(answer_question("How do I fine-tune a BERT model?", "dense_abstention"))
PY
```

Expected out-of-domain MCP response:

```python
{
    "answered": False,
    "answer": "I could not find strong enough supporting evidence in the indexed documentation.",
    "citations": []
}
```

---

## Run Tests

```bash
python -m pytest -v
```

---

