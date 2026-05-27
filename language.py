"""Small language-detection helpers used to choose the answer language."""

from __future__ import annotations


def detect_answer_language(text: str) -> str:
    """Return the response language implied by the user's question text."""
    return "Chinese" if contains_chinese(text) else "English"


def contains_chinese(text: str) -> bool:
    """Check whether the text contains any CJK unified ideograph characters."""
    return any("\u4e00" <= char <= "\u9fff" for char in text)
