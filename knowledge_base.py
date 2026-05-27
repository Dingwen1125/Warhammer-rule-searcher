"""PDF discovery, extraction, optional OCR, and chunking for the rule library."""

from __future__ import annotations

import logging
from pathlib import Path
import re

from pypdf import PdfReader

from config import (
    ENABLE_OCR,
    KNOWLEDGE_BASE_DIR,
    OCR_TEXT_MIN_CHARS,
    SUPPORTED_EXTENSIONS,
)
from nodes.process_images import extract_page_image_text
from state import Chunk


logging.getLogger("pypdf").setLevel(logging.ERROR)


def fetch_documents() -> list[Path]:
    """Fetch local Warhammer rule PDFs from the knowledge-base directory."""
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError("no knowledge_base directory")
    paths = sorted(
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError("no rule pdfs")
    return paths


def ensure_knowledge_base() -> None:
    """Fail early if no rule PDFs are available."""
    fetch_documents()


def preprocess_documents(paths: list[Path]) -> list[Chunk]:
    """Load, OCR when needed, clean, and split rule documents into chunks."""
    chunks: list[Chunk] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            chunks.extend(load_pdf_chunks(path))
    return chunks


def load_pdf_chunks(path: Path) -> list[Chunk]:
    """Extract one PDF into page-aware chunks, using OCR for sparse pages."""
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        original_text = text
        extraction_method = "text"
        if should_ocr_page(text):
            ocr_text = extract_page_image_text(path, page_number)
            if ocr_text:
                text = merge_extracted_text(text, ocr_text)
                extraction_method = "text+ocr" if original_text.strip() else "ocr"
        for chunk_text in split_text(text):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=path.name,
                    page=page_number,
                    document_id=document_id(path),
                    title=document_title(path),
                    extraction_method=extraction_method,
                )
            )
    return chunks


def should_ocr_page(text: str) -> bool:
    """Decide whether extracted text is too sparse and needs vision OCR."""
    return ENABLE_OCR and len(text.strip()) < OCR_TEXT_MIN_CHARS


def merge_extracted_text(text: str, ocr_text: str) -> str:
    """Combine embedded PDF text and OCR text without empty sections."""
    parts = [part.strip() for part in (text, ocr_text) if part.strip()]
    return "\n\n".join(parts)


def document_id(path: Path) -> str:
    """Return a stable knowledge-base-relative identifier for a document."""
    try:
        relative = path.relative_to(KNOWLEDGE_BASE_DIR)
    except ValueError:
        relative = path
    return relative.with_suffix("").as_posix()


def document_title(path: Path) -> str:
    """Convert a file stem into a readable title for citations and scoring."""
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 160) -> list[str]:
    """Split normalized page text into overlapping retrievable chunks."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            sentence_end = max(
                normalized.rfind(". ", start, end),
                normalized.rfind("\n", start, end),
            )
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks
