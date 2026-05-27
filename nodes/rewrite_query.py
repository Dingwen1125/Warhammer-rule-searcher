"""Query rewrite node for producing bilingual Warhammer retrieval queries."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from state import RagState


def rewrite_query(state: RagState) -> RagState:
    """Rewrite the question into a bilingual search query for mixed PDF sources."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    prompt = (
        "Convert the user question into a concise bilingual Warhammer rules "
        "search query for a hybrid search system over multiple Chinese and English "
        "rule PDFs. Keep the user's original important terms, then add likely "
        "official equivalents in the other language when useful. Preserve faction "
        "names, unit names, detachments, stratagems, abilities, weapon names, "
        "keywords, edition names, numbers, and exact rule terms. If the question "
        "names a faction, army, codex, or camp, keep it in the query so the "
        "retriever can choose the right PDF. Return only the search query, not an "
        "answer.\n\n"
        f"Original question: {state['question']}\n"
        f"Current search query: {state.get('search_query', state['question'])}"
    )
    rewritten = llm.invoke(prompt).content
    attempts = state.get("attempts", 0)
    if state.get("chunks"):
        attempts += 1
    return {
        "rewritten_question": str(rewritten).strip(),
        "search_query": str(rewritten).strip(),
        "attempts": attempts,
    }
