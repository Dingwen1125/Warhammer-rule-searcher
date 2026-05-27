"""Custom LangGraph RAG agent over Warhammer rule PDFs.

Usage:
    cp .env.example .env
    # edit .env and set OPENAI_API_KEY
    python3 warhammer_agent.py "Can this unit advance and charge?"
"""

from __future__ import annotations

import argparse
import os

from openai import APIConnectionError, RateLimitError

from env import load_dotenv
from graph import build_graph
from knowledge_base import fetch_documents, preprocess_documents
from retriever import index_documents
from state import RagState


def ask(question: str) -> RagState:
    """Run one question through document loading, retrieval setup, and the graph."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the RAG agent.")
    documents = fetch_documents()
    chunks = preprocess_documents(documents)
    retriever = index_documents(chunks)
    graph = build_graph(retriever)
    return graph.invoke({"question": question, "attempts": 0})


def main() -> None:
    """Parse CLI arguments, run the agent, and print the final answer."""
    parser = argparse.ArgumentParser(description="Ask the Warhammer rules RAG agent.")
    parser.add_argument("question", nargs="?", default="Can this unit advance and charge?")
    args = parser.parse_args()

    try:
        result = ask(args.question)
    except FileNotFoundError as error:
        if str(error) in {"no rule pdfs", "no knowledge_base directory"}:
            raise SystemExit(
                "No Warhammer rule PDFs found. Put one or more .pdf files under "
                "knowledge_base/ and run the command again."
            ) from error
        raise
    except APIConnectionError as error:
        raise SystemExit(
            "Could not connect to the OpenAI API. Check your network/VPN/proxy, "
            "then run the command again."
        ) from error
    except RateLimitError as error:
        message = str(error)
        if "insufficient_quota" in message:
            raise SystemExit(
                "OpenAI API quota is insufficient. Check your API key billing/quota, "
                "then run the command again."
            ) from error
        raise
    print(result["answer"])


if __name__ == "__main__":
    main()
