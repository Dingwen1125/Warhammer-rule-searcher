from __future__ import annotations

from state import RagState


def no_answer(state: RagState) -> RagState:
    return {
        "answer": (
            "I could not find relevant information in the PDF knowledge base, "
            "so I will not answer this question."
        )
    }
