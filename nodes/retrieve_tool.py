"""LangGraph node wrapper around the hybrid Warhammer rules retriever."""

from __future__ import annotations

from retriever import InMemoryRetriever, create_retriever_tool
from state import RagState


def make_retrieve_tool_node(retriever: InMemoryRetriever):
    """Bind a prepared retriever instance into a LangGraph node function."""
    retriever_tool = create_retriever_tool(retriever)

    def retrieve_tool(state: RagState) -> RagState:
        """Run retrieval and store returned chunks in graph state."""
        # Prefer the bilingual rewritten query, then fall back to the raw question.
        query = state.get("rewritten_question") or state.get("search_query") or state["question"]
        return {"chunks": retriever_tool(query), "attempts": state.get("attempts", 0)}

    return retrieve_tool
