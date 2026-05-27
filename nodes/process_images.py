from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import CHAT_MODEL, OCR_CACHE_PATH, OCR_MODEL


def extract_page_image_text(path: Path, page_number: int) -> str:
    """Render an image-heavy PDF page and extract visible rules text."""
    cache = load_ocr_cache(OCR_CACHE_PATH)
    key = ocr_cache_key(path, page_number)
    cached_text = cache.setdefault("entries", {}).get(key)
    if isinstance(cached_text, str):
        return cached_text

    image_bytes = render_pdf_page(path, page_number)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    llm = ChatOpenAI(model=OCR_MODEL or CHAT_MODEL, temperature=0.0)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract all visible Warhammer rules text from this PDF page "
                    "image, whether it is Chinese, English, or mixed. Preserve "
                    "headings, bullet points, stat lines, keywords, ability names, "
                    "numbers, symbols, and short table rows exactly as shown. Return "
                    "only extracted text."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
    )
    text = str(llm.invoke([message]).content).strip()
    cache["entries"][key] = text
    save_ocr_cache(OCR_CACHE_PATH, cache)
    return text


def load_ocr_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if raw.get("version") != 1 or not isinstance(raw.get("entries"), dict):
        return {"version": 1, "entries": {}}
    return raw


def save_ocr_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def ocr_cache_key(path: Path, page_number: int) -> str:
    stat = path.stat()
    payload = {
        "model": OCR_MODEL or CHAT_MODEL,
        "path": str(path.resolve()),
        "page": page_number,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def render_pdf_page(path: Path, page_number: int) -> bytes:
    try:
        import fitz  # PyMuPDF
    except ImportError as error:
        raise RuntimeError(
            "OCR is enabled, but PyMuPDF is not installed. Install requirements "
            "or set WARHAMMER_ENABLE_OCR=0."
        ) from error

    with fitz.open(path) as document:
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return pixmap.tobytes("png")
