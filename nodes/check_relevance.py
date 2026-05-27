from __future__ import annotations

import math
from typing import Literal

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from retriever import format_context, meaningful_terms, tokenize
from state import RagState


def check_relevance(state: RagState) -> RagState:
    if has_unmatched_requested_source(state):
        return {
            "relevant_chunks": [],
            "relevance_reason": (
                "NO\nThe question appears to request a specific source/faction, "
                "but none of the retrieved source names match it."
            ),
        }

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    context = format_context(state.get("chunks", []))
    prompt = f"""
You are checking whether retrieved PDF chunks are relevant to a Warhammer rules question.
If the question names or implies a specific faction, army, codex, index, unit,
or rules source, the chunks are relevant only when the retrieved source or text
actually matches that requested source. Do not treat a different faction/codex
as a substitute just because it has a similar rule structure.
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


def has_unmatched_requested_source(state: RagState) -> bool:
    query_text = " ".join(
        value
        for value in (
            state.get("rewritten_question"),
            state.get("search_query"),
            state.get("question"),
        )
        if value
    )
    question_terms = set(meaningful_terms(tokenize(query_text)))
    if not question_terms:
        return False
    chunks = state.get("chunks", [])
    if not chunks:
        return False
    best_source_score = max((chunk.source_score for chunk in chunks), default=0.0)
    best_keyword_score = max((chunk.keyword_score for chunk in chunks), default=0.0)
    if best_source_score > 0:
        return False
    source_text = " ".join(f"{chunk.document_id} {chunk.source} {chunk.title}" for chunk in chunks)
    source_terms = set(meaningful_terms(tokenize(source_text)))
    missing_source_terms = question_terms - source_terms
    return bool(missing_source_terms) and best_keyword_score < 0.5


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
