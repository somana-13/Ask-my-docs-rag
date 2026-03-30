# Ask My Docs RAG

A domain-specific retrieval system over technical documentation that returns **grounded evidence with citations** instead of relying only on model memory.

## What it does

This project indexes a technical docs corpus and supports evidence-based question answering through:

- markdown document ingestion
- section-aware chunking with overlap
- semantic retrieval using embeddings
- BM25 keyword retrieval
- hybrid retrieval (dense + BM25)
- cross-encoder reranking
- abstention when evidence is weak
- citation-aware evidence display

## Corpus

This version uses a curated subset of the **HTTPX documentation**.

## Tech Stack

- Python
- Sentence Transformers
- ChromaDB
- LangChain
- BM25 / `rank-bm25`
- Pytest

## Current Features

### Phase 1
- ingest markdown docs
- split into metadata-rich chunks
- build persistent vector index
- retrieve semantically relevant evidence
- display source-aware citations

### Phase 2
- add BM25 lexical retrieval
- combine dense + keyword retrieval
- rerank hybrid candidates with a cross-encoder
- filter weak results
- abstain when strong supporting evidence is missing

## Project Structure

```text
ask-my-docs-rag/
├── data/
│   ├── raw/httpx/                 # Source docs corpus
│   └── processed/
│       ├── ingested_docs.json     # Parsed docs with sections
│       └── chunks.json            # Retrieval-ready chunks
├── scripts/
│   ├── ingest_docs.py             # Parse raw markdown docs
│   ├── build_chunks.py            # Create overlapping chunks
│   ├── build_index.py             # Embed chunks and store in Chroma
│   ├── query_index.py             # Dense retrieval
│   ├── query_bm25.py              # BM25 retrieval
│   ├── query_hybrid.py            # Dense + BM25 retrieval
│   └── query_reranked.py          # Hybrid retrieval + reranking + abstention
├── src/
│   ├── ingestion/
│   │   └── loaders.py
│   ├── chunking/
│   │   └── splitter.py
│   ├── embeddings/
│   │   └── embedder.py
│   └── retrieval/
│       ├── vector_store.py
│       ├── bm25_retriever.py
│       ├── hybrid_retriever.py
│       └── reranker.py
├── tests/
│   └── test_pipeline.py
├── chroma_db/                     # Local vector store
├── pytest.ini
├── .gitignore
└── README.md

How it works
	1.	Raw markdown docs are ingested and split by headings.
	2.	Each section is chunked into overlapping retrieval units.
	3.	Chunks are embedded and stored in ChromaDB.
	4.	Queries can be answered through:
	•	dense retrieval
	•	BM25 retrieval
	•	hybrid retrieval
	5.	Hybrid candidates are reranked with a cross-encoder.
	6.	If supporting evidence is too weak, the system abstains.

How to run

1. Ingest documents
   python -m scripts.ingest_docs

2. Build chunks
   python -m scripts.build_chunks

3. Build vector index
   python -m scripts.build_index

4. Run dense retrieval
   python -m scripts.query_index

5. Run BM25 retrieval
   python -m scripts.query_bm25

6. Run hybrid retrieval
   python -m scripts.query_hybrid

7. Run hybrid retrieval + reranking + abstention
   python -m scripts.query_reranked

8. Run tests
   python -m pytest -v


Example queries
	•	How do I configure authentication in HTTPX?
	•	DigestAuth
	•	NetRCAuth
	•	How do timeouts work in HTTPX?

Example abstention

For questions outside the indexed corpus, the system refuses to answer:

I could not find strong enough supporting evidence in the indexed documents.

### Why this project matters

This project demonstrates practical RAG system design:
	•	retrieval over real documentation
	•	metadata-aware chunking
	•	hybrid search
	•	reranking for better precision
	•	trust-oriented abstention behavior
	•	test-backed preprocessing reliability

Next Step

Phase 3 will add:
	•	a golden evaluation dataset
	•	faithfulness checks
	•	automated evaluation in CI

