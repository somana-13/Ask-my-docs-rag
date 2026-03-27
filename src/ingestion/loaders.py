from pathlib import Path
import re


def read_markdown_file(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def extract_doc_id(file_path: str) -> str:
    path = Path(file_path)
    relative = str(path).replace("\\", "/")
    name = relative.replace("/", "_").replace(".md", "")
    return name


def split_markdown_by_headings(text: str):
    lines = text.splitlines()
    sections = []
    current_title = "Introduction"
    current_lines = []

    heading_pattern = re.compile(r"^(#{1,3})\s+(.*)$")

    for line in lines:
        match = heading_pattern.match(line.strip())
        if match:
            if current_lines:
                sections.append(
                    {
                        "section_title": current_title,
                        "section_text": "\n".join(current_lines).strip(),
                    }
                )
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section_title": current_title,
                "section_text": "\n".join(current_lines).strip(),
            }
        )

    return sections


def load_markdown_document(file_path: str):
    text = read_markdown_file(file_path)
    doc_id = extract_doc_id(file_path)
    sections = split_markdown_by_headings(text)

    return {
        "doc_id": doc_id,
        "source_name": Path(file_path).name,
        "source_path": str(Path(file_path).as_posix()),
        "sections": sections,
    } 