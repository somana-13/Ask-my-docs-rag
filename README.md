# Ask My Docs RAG

A domain-specific “ask my docs” retrieval system built over technical documentation, designed to ground every response in retrieved evidence and surface clear citations.

## Project Overview

How It Works:

1. Raw markdown technical docs are ingested and split into sections.
2. Sections are chunked into overlapping retrieval units with metadata preserved.
3. Each chunk is embedded using Sentence Transformers.
4. Embeddings and metadata are stored in a persistent Chroma vector database.
5. User queries are embedded and matched against the indexed chunks.
6. The system returns the most relevant evidence with source-aware citations.

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

----------------------------------------------------------------------------------------------------
## Example Retrieval Output

### Query
How do I configure authentication in HTTPX?

### Retrieved Evidence
TOP RETRIEVED EVIDENCE:
1. authentication.md | NetRC authentication
2. authentication.md | Introduction
3. quickstart.md | Authentication

### Evidence based summary
- [authentication.md | NetRC authentication] HTTPX can be configured to use a .netrc file for authentication.
- [authentication.md | Introduction] Authentication can be included on a per-request basis or configured on the client.
- [quickstart.md | Authentication] HTTPX supports Basic and Digest HTTP authentication.

-----------------------------------------------------------------------------------------------------
Command To Run

(.venv) somana@Mac ask-my-docs-rag % python -m scripts.query_index

-----------------------------------------------------------------------------------------------------
INPUT
Enter your question: How do I configure authentication in HTTPX?

OUTPUT

TOP RETRIEVED EVIDENCE:
1. authentication.md | NetRC authentication (chunk_id=data_raw_httpx_advanced_authentication_chunk_003, distance=0.7308)
2. authentication.md | Introduction (chunk_id=data_raw_httpx_advanced_authentication_chunk_000, distance=0.7662)
3. quickstart.md | Authentication (chunk_id=data_raw_httpx_quickstart_chunk_017, distance=0.7681)

EVIDENCE-BASED SUMMARY:
- [authentication.md | NetRC authentication] HTTPX can be configured to use [a `.netrc` config file](https://everything.curl.dev/usingcurl/netrc) for authentication.  The `.netrc` config file allows authentication credentials to be associated with specified hosts....
- [authentication.md | Introduction] Authentication can either be included on a per-request basis...  ```pycon >>> auth = httpx.BasicAuth(username="username", password="secret") >>> client = httpx.Client() >>> response = client.get("https://www.example.com/...
- [quickstart.md | Authentication] HTTPX supports Basic and Digest HTTP authentication.  To provide Basic authentication credentials, pass a 2-tuple of plaintext `str` or `bytes` objects as the `auth` argument to the request functions:  ```pycon >>> httpx...

SOURCES:
- authentication.md | NetRC authentication
- authentication.md | Introduction
- quickstart.md | Authentication

----------------------------------------------------------------------------------------------------
RAW TOP CHUNK:
HTTPX can be configured to use [a `.netrc` config file](https://everything.curl.dev/usingcurl/netrc) for authentication.

The `.netrc` config file allows authentication credentials to be associated with specified hosts. When a request is made to a host that is found in the netrc file, the username and password will be included using HTTP basic authentication.

Example `.netrc` file:

```
machine example.org
login example-username
password example-password

machine python-httpx.org
login other-username
password other-password
```

Some examples of configuring `.netrc` authentication with `httpx`.

Use the default `.netrc` file in the users home directory:

```pycon
>>> auth = httpx.NetRCAuth()
>>> client = httpx.Client(auth=auth)
```

Use an explicit path to a `.netrc` file:

```pycon
>>> auth = httpx.NetRCAuth(file="/path/to/.netrc")
>>> client = httpx.Client(auth=auth)
```

Use the `NETRC` environment variable to configure a path to the `.netrc` file,
or fallback to the default.

```pycon
>>> auth = httpx.NetRCAuth(file=os.environ.get("NETRC"))
>>> client = httpx.Client(auth=auth)
```

The `NetRCAuth()` class uses [the `netrc.netrc()` function from the Python standard library](https://docs.python.org/3/library/netrc.html). See the documentation there for more details on exceptions that may be raised if the `.netrc` file is not found, or cannot be parsed.