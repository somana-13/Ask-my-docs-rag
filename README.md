# Ask My Docs RAG

A production-style Retrieval-Augmented Generation (RAG) system over technical documentation. The project retrieves grounded evidence with citations, compares retrieval variants using MLflow, simulates A/B testing for system changes, and exposes the RAG pipeline through MCP tools for agent integration.

## Project Summary

This project started as a documentation search system and was upgraded into an end-to-end ML/AI evaluation platform.

It includes:

- document ingestion and section-aware chunking
- dense vector retrieval with Sentence Transformers and ChromaDB
- BM25 keyword retrieval
- hybrid retrieval with cross-encoder reranking
- evidence-based abstention for weak or out-of-domain queries
- MLflow experiment tracking
- A/B testing simulation
- MCP server tools for agent-ready RAG access

## Corpus

This version indexes a curated subset of the **HTTPX documentation**.

Indexed sources include:

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

## Tech Stack

- Python
- Sentence Transformers
- ChromaDB
- LangChain
- BM25 / `rank-bm25`
- Cross-encoder reranking
- MLflow
- Pandas / SciPy
- MCP
- Pytest

## Features

### Phase 1: RAG Baseline

- ingest markdown documentation
- split documents into section-aware chunks
- preserve metadata such as source file, section title, and chunk id
- build a persistent ChromaDB vector index
- run dense semantic retrieval
- return source-aware evidence snippets

### Phase 2: Better Retrieval

- add BM25 lexical retrieval
- combine dense and BM25 results using hybrid retrieval
- rerank candidates using a cross-encoder
- filter weak evidence
- abstain when retrieved evidence is not strong enough

### Phase 3: MLflow Experiment Tracking

- define config-driven retrieval experiments
- track retriever type, top-k, reranker settings, and abstention settings
- log retrieval metrics, latency, and error rate
- save prediction CSVs and summary JSON artifacts
- generate an experiment comparison report

### Phase 4: A/B Testing Simulation

- compare a control system against a treatment system
- Variant A: dense retrieval baseline
- Variant B: dense retrieval with evidence-based abstention
- measure in-domain source match, keyword match, out-of-domain abstention, latency, and error rate
- generate an A/B test report

### Phase 5: MCP Tool Server

- expose the RAG pipeline as MCP tools
- allow an AI agent to call documentation search and grounded answering tools
- support out-of-domain refusal behavior through the default `dense_abstention` variant

## Project Structure

```text
ask-my-docs-rag/
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
│   └── ab_test_report.md
├── scripts/
│   ├── ingest_docs.py
│   ├── build_chunks.py
│   ├── build_index.py
│   ├── query_index.py
│   ├── query_bm25.py
│   ├── query_hybrid.py
│   └── query_reranked.py
├── src/
│   ├── chunking/
│   ├── embeddings/
│   ├── ingestion/
│   └── retrieval/
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Setup

```bash
cd ask-my-docs-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build the RAG Index

```bash
python -m scripts.ingest_docs
python -m scripts.build_chunks
python -m scripts.build_index
```

## Run Retrieval Manually

```bash
python -m scripts.query_index
python -m scripts.query_bm25
python -m scripts.query_hybrid
python -m scripts.query_reranked
```

Example queries:

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

## Run Tests

```bash
python -m pytest -v
```

