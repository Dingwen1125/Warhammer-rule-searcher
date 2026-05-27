from __future__ import annotations

import math
from typing import Literal

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from retriever import format_context
from state import RagState


def check_relevance(state: RagState) -> RagState:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    context = format_context(state.get("chunks", []))
    prompt = f"""
You are checking whether retrieved PDF chunks are relevant to a Warhammer rules question.
Return the first line as YES or NO. On the second line, give a short reason.

Question: {state["question"]}

Retrieved chunks:
{context}
"""
    raw = str(llm.invoke(prompt).content).strip()
    first_line = raw.splitlines()[0].upper() if raw else "NO"
    is_relevant = "YES" in first_line
    return {
        "relevant_chunks": state.get("chunks", []) if is_relevant else [],
        "relevance_reason": raw,
    }


def relevance_route(state: RagState) -> Literal["generate", "rewrite", "no_answer"]:
    attempts = state.get("attempts", 0)
    best_score = max((chunk.score for chunk in state.get("chunks", [])), default=-math.inf)
    has_relevant_chunks = bool(state.get("relevant_chunks"))
    found_relevant_content = has_relevant_chunks and best_score >= 0.12
    if found_relevant_content:
        return "generate"
    if attempts < 2:
        return "rewrite"
    return "no_answer"
