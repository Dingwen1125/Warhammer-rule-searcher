"""Terminal node used when the knowledge base cannot answer the question."""

from __future__ import annotations

from state import RagState


def no_answer(state: RagState) -> RagState:
    """Return the standard refusal when retrieved context is not relevant."""
    return {
        "answer": (
            "I could not find relevant information in the PDF knowledge base, "
            "so I will not answer this question."
        )
    }
