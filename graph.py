"""LangGraph workflow assembly for the Warhammer rules agent."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes.agent import agent, should_retrieve
from nodes.check_relevance import check_relevance, relevance_route
from nodes.generate import generate
from nodes.no_answer import no_answer
from nodes.prepare_query import prepare_query
from nodes.retrieve_tool import make_retrieve_tool_node
from nodes.rewrite_query import rewrite_query
from retriever import InMemoryRetriever
from state import RagState


def build_graph(retriever: InMemoryRetriever):
    """Compile the retrieval, grading, rewrite, and answer-generation workflow."""
    graph = StateGraph(RagState)
    graph.add_node("prepare_query", prepare_query)
    graph.add_node("agent", agent)
    graph.add_node("retrieve_tool", make_retrieve_tool_node(retriever))
    graph.add_node("check_relevance", check_relevance)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "prepare_query")
    graph.add_edge("prepare_query", "rewrite_query")
    graph.add_edge("rewrite_query", "agent")
    graph.add_conditional_edges(
        "agent",
        should_retrieve,
        {"continue": "retrieve_tool", "end": END},
    )
    graph.add_edge("retrieve_tool", "check_relevance")
    graph.add_conditional_edges(
        "check_relevance",
        relevance_route,
        {"rewrite": "rewrite_query", "generate": "generate", "no_answer": "no_answer"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)
    return graph.compile()
