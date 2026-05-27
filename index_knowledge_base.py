"""Preprocess rule PDFs and cache embeddings for new or changed chunks."""

from __future__ import annotations

import os

from openai import APIConnectionError, RateLimitError

from config import EMBEDDING_CACHE_PATH
from env import load_dotenv
from knowledge_base import fetch_documents, preprocess_documents
from retriever import embedding_cache_status, index_documents


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before indexing the knowledge base.")

    try:
        documents = fetch_documents()
        chunks = preprocess_documents(documents)
        total, cached, missing = embedding_cache_status(chunks)
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {total}")
        print(f"Cached embeddings: {cached}")
        print(f"New embeddings needed: {missing}")

        if missing:
            retriever = index_documents(chunks)
            retriever.build_index()
            print(f"Saved embedding cache: {EMBEDDING_CACHE_PATH}")
        else:
            print("Embedding cache is already up to date.")
    except FileNotFoundError as error:
        if str(error) in {"no rule pdfs", "no knowledge_base directory"}:
            raise SystemExit(
                "No Warhammer rule PDFs found. Put one or more .pdf files under "
                "knowledge_base/ and run this command again."
            ) from error
        raise
    except APIConnectionError as error:
        raise SystemExit(
            "Could not connect to the OpenAI API while embedding. Check your "
            "network/VPN/proxy, then run the command again."
        ) from error
    except RateLimitError as error:
        message = str(error)
        if "insufficient_quota" in message:
            raise SystemExit(
                "OpenAI API quota is insufficient. Check your API key billing/quota, "
                "then run the command again."
            ) from error
        raise


if __name__ == "__main__":
    main()
