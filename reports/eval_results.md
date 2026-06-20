# Agentic RAG Evaluation Results

## Setup

Evaluated the LangGraph-based Agentic RAG pipeline on an initial 10-question golden dataset covering:

- Simple lookup questions
- Multi-hop questions
- Comparison questions
- Clarification / unanswerable questions

The system uses:

- LangGraph routing
- Multi-hop decomposition
- Existing hybrid/dense retrieval pipeline
- Citation tracking
- Local NLI cross-encoder faithfulness scoring

## Final Metrics

| Metric | Score |
|---|---:|
| Router accuracy | 1.0000 |
| Retrieval precision@5 | 1.0000 |
| Citation coverage | 1.0000 |
| Abstention accuracy | 1.0000 |
| Faithfulness score | 0.9500 |
| Average latency | 175.57 ms |

## Improvement After Failure Analysis

| Metric | Before | After |
|---|---:|---:|
| Router accuracy | 0.8000 | 1.0000 |
| Retrieval precision@5 | 0.9500 | 1.0000 |
| Citation coverage | 0.8833 | 1.0000 |
| Abstention accuracy | 0.8000 | 1.0000 |
| Faithfulness score | 0.7367 | 0.9500 |
| Average latency | 270.91 ms | 175.57 ms |

## Key Fixes

- Added punctuation-normalized routing for vague clarification queries.
- Added out-of-corpus detection for cloud/deployment questions.
- Cleaned retrieved markdown before answer generation.
- Improved claim extraction to remove answer-template prefixes before NLI scoring.
- Removed incomplete markdown/list fragments from faithfulness evaluation.

## Main Takeaway

The upgraded system demonstrates evaluation-driven Agentic RAG development: routing, multi-hop decomposition, citation-aware retrieval, local NLI faithfulness scoring, and failure-analysis-driven improvements.
