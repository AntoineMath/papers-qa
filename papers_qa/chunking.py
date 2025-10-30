import os
import re
from typing import List, Tuple
from PyPDF2 import PdfReader
import tiktoken

SECTION_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-]+")


def pdf_to_token_chunks(
    folder: str, chunk_size: int, overlap: int, model_encoding: str = "cl100k_base"
) -> List[Tuple[str, str, str]]:
    enc = tiktoken.get_encoding(model_encoding)
    items: List[Tuple[str, str, str]] = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, name)
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"Warning: Failed to read PDF {path}: {e}")
            continue

        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]

        section_markers: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            if SECTION_PATTERN.match(line):
                section_markers.append((i, line.strip()))

        tokens = enc.encode("\n".join(lines))
        start = 0
        while start < len(tokens):
            chunk_tokens = tokens[start:start + chunk_size]
            chunk_text = enc.decode(chunk_tokens).strip()
            if not chunk_text:
                start += chunk_size - overlap
                continue

            text_before_chunk = enc.decode(tokens[:start])
            approx_lines_before = text_before_chunk.count('\n')
            chunk_section = "Introduction"
            for line_idx, sect_title in section_markers:
                if line_idx <= approx_lines_before:
                    chunk_section = sect_title
                else:
                    break

            items.append((name, chunk_section, chunk_text))
            start += chunk_size - overlap

    return items


