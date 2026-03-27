from typing import List, Dict


def split_text_with_overlap(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping character-based chunks.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks


def chunk_document_sections(doc: Dict, chunk_size: int = 1500, overlap: int = 200) -> List[Dict]:
    """
    Takes one ingested doc and converts its sections into retrieval chunks.
    Preserves source metadata.
    """
    chunked_docs = []
    chunk_counter = 0

    for section in doc.get("sections", []):
        section_title = section.get("section_title", "Unknown Section")
        section_text = section.get("section_text", "").strip()

        if not section_text:
            continue

        text_chunks = split_text_with_overlap(
            text=section_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in text_chunks:
            chunk = {
                "doc_id": doc["doc_id"],
                "source_name": doc["source_name"],
                "source_path": doc["source_path"],
                "section_title": section_title,
                "chunk_id": f"{doc['doc_id']}_chunk_{chunk_counter:03d}",
                "chunk_index": chunk_counter,
                "text": chunk_text,
                "char_count": len(chunk_text),
            }
            chunked_docs.append(chunk)
            chunk_counter += 1

    return chunked_docs