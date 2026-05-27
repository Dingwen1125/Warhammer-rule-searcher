from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import CHAT_MODEL, OCR_MODEL


def extract_page_image_text(path: Path, page_number: int) -> str:
    """Render an image-heavy PDF page and extract visible rules text."""
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
    return str(llm.invoke([message]).content).strip()


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
