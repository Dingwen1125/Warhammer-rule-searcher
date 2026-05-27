from __future__ import annotations

import os
from pathlib import Path


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
EMBEDDING_CACHE_PATH = Path(".cache/warhammer_embeddings.json")
OCR_CACHE_PATH = Path(".cache/warhammer_ocr.json")
SUPPORTED_EXTENSIONS = {".pdf"}
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", CHAT_MODEL)
ENABLE_OCR = os.getenv("WARHAMMER_ENABLE_OCR", "0").lower() not in {"0", "false", "no"}
OCR_TEXT_MIN_CHARS = int(os.getenv("WARHAMMER_OCR_TEXT_MIN_CHARS", "80"))
