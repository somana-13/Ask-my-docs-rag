from src.ingestion.loaders import split_markdown_by_headings
from src.chunking.splitter import split_text_with_overlap, chunk_document_sections


def test_split_markdown_by_headings():
    markdown_text = """
# Introduction
This is the intro section.

## Authentication
This section explains authentication.

## Timeouts
This section explains timeouts.
"""

    sections = split_markdown_by_headings(markdown_text)

    assert len(sections) == 3
    assert sections[0]["section_title"] == "Introduction"
    assert "intro section" in sections[0]["section_text"]
    assert sections[1]["section_title"] == "Authentication"
    assert sections[2]["section_title"] == "Timeouts"


def test_split_text_with_overlap_creates_multiple_chunks():
    text = "A" * 4000
    chunks = split_text_with_overlap(text, chunk_size=1500, overlap=200)

    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunk_document_sections_preserves_metadata():
    doc = {
        "doc_id": "test_doc",
        "source_name": "test.md",
        "source_path": "data/raw/test.md",
        "sections": [
            {
                "section_title": "Authentication",
                "section_text": "A" * 2500
            }
        ]
    }

    chunks = chunk_document_sections(doc, chunk_size=1000, overlap=100)

    assert len(chunks) >= 2

    first_chunk = chunks[0]

    assert first_chunk["doc_id"] == "test_doc"
    assert first_chunk["source_name"] == "test.md"
    assert first_chunk["source_path"] == "data/raw/test.md"
    assert first_chunk["section_title"] == "Authentication"
    assert "chunk_id" in first_chunk
    assert "chunk_index" in first_chunk
    assert "text" in first_chunk
    assert "char_count" in first_chunk