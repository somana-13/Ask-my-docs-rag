# Ask My Docs RAG

A domain-specific “ask my docs” retrieval system built over technical documentation, designed to ground every response in retrieved evidence and surface clear citations.

## Project Goal

The goal of this project is to build a reliable documentation QA system that answers questions using retrieved evidence from a document corpus instead of relying only on model memory.

This project is being developed in phases:

- **Phase 1:** ingestion, chunking, embeddings, vector indexing, semantic retrieval, and citation-aware evidence display
- **Phase 2:** hybrid retrieval, reranking, and explicit abstention
- **Phase 3:** evaluation pipeline, golden dataset, and CI-based quality checks

## Current Status

### Phase 1 Complete
The current system supports:

- ingesting markdown technical documentation
- splitting documents into section-aware chunks
- preserving metadata for traceability and citation
- embedding chunks using Sentence Transformers
- indexing embeddings in a persistent Chroma vector store
- retrieving semantically relevant chunks for a user query
- displaying retrieved evidence with source and section citations

## Corpus

This Phase 1 version uses a curated subset of the **HTTPX documentation** as the technical document corpus.

## Tech Stack

- **Python**
- **Sentence Transformers** for embeddings
- **ChromaDB** for persistent vector storage
- **Markdown-based ingestion pipeline**
- **Terminal-based query interface**

## Project Structure

```text
ask-my-docs-rag/
├── data/
│   ├── raw/httpx/                 # Source technical docs corpus
│   └── processed/
│       ├── ingested_docs.json     # Parsed docs with section structure
│       └── chunks.json            # Retrieval-ready chunks with metadata
├── chroma_db/                     # Local persistent Chroma vector store
├── scripts/
│   ├── ingest_docs.py             # Load raw markdown docs
│   ├── build_chunks.py            # Section-aware chunking
│   ├── build_index.py             # Generate embeddings and store in Chroma
│   └── query_index.py             # Query the vector index and display evidence
├── src/
│   ├── ingestion/
│   │   └── loaders.py
│   ├── chunking/
│   │   └── splitter.py
│   ├── embeddings/
│   │   └── embedder.py
│   └── retrieval/
│       └── vector_store.py
└── tests/

Pipeline Overview

1. Document Ingestion

Raw markdown documentation files are loaded and parsed into structured documents.
Each document is split into sections using markdown headings.

2. Section-Aware Chunking

Each section is split into smaller overlapping chunks to make retrieval more precise while preserving context near boundaries.

Current strategy:
	•	character-based chunking
	•	overlap between adjacent chunks
	•	metadata preserved for every chunk

3. Embedding Generation

Each chunk is converted into a dense vector embedding using a Sentence Transformers model.

To improve retrieval quality for technical docs, embeddings are generated from enriched chunk text that includes:
	•	source filename
	•	section title
	•	chunk content

4. Vector Indexing

Embeddings, chunk text, and metadata are stored in a persistent Chroma collection for semantic search.

5. Retrieval

Given a user query:
	•	the query is embedded
	•	the vector store retrieves the most relevant chunks
	•	retrieved evidence is displayed with source and section citations

Example Query

Question:
How do I configure authentication in HTTPX?

Retrieved Evidence Includes:
	•	authentication.md | Introduction
	•	quickstart.md | Authentication
	•	authentication.md | NetRC authentication
