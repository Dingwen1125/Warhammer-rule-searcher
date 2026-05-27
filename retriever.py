"""Fast hybrid retriever with cached embeddings and bilingual keyword scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_CACHE_PATH, EMBEDDING_MODEL
from state import Chunk


class InMemoryRetriever:
    """In-memory hybrid retriever over cached document embeddings."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """Store chunks and prepare local keyword statistics for retrieval."""
        if not chunks:
            raise ValueError("No text chunks were loaded from the PDF.")
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.matrix: np.ndarray | None = None
        self.tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        self.document_frequencies = Counter(
            token for tokens in self.tokenized_chunks for token in set(tokens)
        )

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        """Return the top chunks using vector, keyword, and source-name scores."""
        self._ensure_index()
        query_vector = self._normalize(
            np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        )[0]
        if self.matrix is None:
            raise RuntimeError("The document index was not created.")
        vector_scores = self.matrix @ query_vector
        keyword_scores = np.array(
            [self._keyword_score(query, index) for index in range(len(self.chunks))],
            dtype=np.float32,
        )
        source_scores = np.array(
            [self._source_score(query, index) for index in range(len(self.chunks))],
            dtype=np.float32,
        )
        scores = (0.72 * normalize_scores(vector_scores)) + (
            0.28 * normalize_scores(keyword_scores)
        ) + (
            0.35 * normalize_scores(source_scores)
        )
        indexes = np.argsort(scores)[::-1][:k]
        return [
            Chunk(
                text=self.chunks[index].text,
                source=self.chunks[index].source,
                page=self.chunks[index].page,
                document_id=self.chunks[index].document_id,
                title=self.chunks[index].title,
                extraction_method=self.chunks[index].extraction_method,
                score=float(scores[index]),
                vector_score=float(vector_scores[index]),
                keyword_score=float(keyword_scores[index]),
                source_score=float(source_scores[index]),
            )
            for index in indexes
        ]

    def _keyword_score(self, query: str, index: int) -> float:
        """Score one chunk by bilingual token overlap with IDF weighting."""
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        chunk_terms = Counter(self.tokenized_chunks[index])
        score = 0.0
        total_chunks = len(self.chunks)
        for term in query_terms:
            if term not in chunk_terms:
                continue
            idf = math.log((1 + total_chunks) / (1 + self.document_frequencies[term])) + 1.0
            score += idf * (1 + math.log(chunk_terms[term]))
        return score / max(1, len(query_terms))

    def _source_score(self, query: str, index: int) -> float:
        """Score whether the query terms match the chunk's file/title metadata."""
        query_terms = meaningful_terms(tokenize(query))
        if not query_terms:
            return 0.0
        chunk = self.chunks[index]
        source_terms = set(
            meaningful_terms(tokenize(f"{chunk.document_id} {chunk.source} {chunk.title}"))
        )
        if not source_terms:
            return 0.0
        matched_terms = [term for term in query_terms if term in source_terms]
        return len(matched_terms) / len(query_terms)

    def _ensure_index(self) -> None:
        """Load cached vectors or create missing embeddings before searching."""
        if self.matrix is not None:
            return
        vectors = load_or_create_embeddings(self.chunks, self.embeddings)
        self.matrix = self._normalize(np.array(vectors, dtype=np.float32))

    def build_index(self) -> None:
        """Create and cache document embeddings without running a search."""
        self._ensure_index()

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """Normalize vectors so dot products behave like cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def index_documents(chunks: list[Chunk]) -> InMemoryRetriever:
    """Prepare a lazy semantic-search index over preprocessed chunks.

    Embeddings are created on the first retrieval call, so questions that do not
    need the retriever tool do not pay the indexing cost.
    """
    return InMemoryRetriever(chunks)


def create_retriever_tool(retriever: InMemoryRetriever):
    """Create the retrieval tool that the LangGraph agent workflow can call."""

    def retrieve_warhammer_rules(query: str, k: int = 6) -> list[Chunk]:
        """Hybrid-search the Warhammer rules knowledge base for relevant chunks."""
        return retriever.search(query, k=k)

    retrieve_warhammer_rules.__name__ = "retrieve_warhammer_rules"
    return retrieve_warhammer_rules


def load_or_create_embeddings(chunks: list[Chunk], embeddings: OpenAIEmbeddings) -> list[list[float]]:
    """Return vectors for chunks, embedding only cache misses."""
    cache = load_embedding_cache(EMBEDDING_CACHE_PATH)
    entries: dict[str, list[float]] = cache.setdefault("entries", {})
    keys = [embedding_cache_key(chunk) for chunk in chunks]
    missing_indexes = [index for index, key in enumerate(keys) if key not in entries]
    if missing_indexes:
        missing_vectors = embeddings.embed_documents(
            [chunks[index].text for index in missing_indexes]
        )
        for index, vector in zip(missing_indexes, missing_vectors, strict=True):
            entries[keys[index]] = vector
        save_embedding_cache(EMBEDDING_CACHE_PATH, cache)
    return [entries[key] for key in keys]


def embedding_cache_status(chunks: list[Chunk]) -> tuple[int, int, int]:
    """Return total, cached, and missing embedding counts for a chunk list."""
    cache = load_embedding_cache(EMBEDDING_CACHE_PATH)
    entries: dict[str, list[float]] = cache.setdefault("entries", {})
    keys = [embedding_cache_key(chunk) for chunk in chunks]
    cached = sum(1 for key in keys if key in entries)
    missing = len(keys) - cached
    return len(keys), cached, missing


def load_embedding_cache(path: Path) -> dict[str, Any]:
    """Read the JSON embedding cache, falling back to an empty cache if invalid."""
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if raw.get("version") != 1 or not isinstance(raw.get("entries"), dict):
        return {"version": 1, "entries": {}}
    return raw


def save_embedding_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write the embedding cache JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def embedding_cache_key(chunk: Chunk) -> str:
    """Build a stable cache key that changes when chunk text or metadata changes."""
    payload = {
        "model": EMBEDDING_MODEL,
        "document_id": chunk.document_id,
        "source": chunk.source,
        "page": chunk.page,
        "extraction_method": chunk.extraction_method,
        "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def format_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks with scores and citations for LLM prompts."""
    return "\n\n".join(
        (
            f"[{idx}] {chunk.title} ({chunk.source}), page {chunk.page}, "
            f"score {chunk.score:.3f}, vector {chunk.vector_score:.3f}, "
            f"keyword {chunk.keyword_score:.3f}, source {chunk.source_score:.3f}, "
            f"extracted {chunk.extraction_method}\n"
            f"{chunk.text}"
        )
        for idx, chunk in enumerate(chunks, start=1)
    )


def tokenize(text: str) -> list[str]:
    """Tokenize mixed English/Chinese text for keyword retrieval."""
    tokens = re.findall(r"[a-z0-9][a-z0-9'_-]*", text.lower())
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in chinese_runs:
        tokens.append(run)
        tokens.extend(character_ngrams(run, 2))
        tokens.extend(character_ngrams(run, 3))
    return tokens


def character_ngrams(text: str, size: int) -> list[str]:
    """Create fixed-size character n-grams for Chinese keyword matching."""
    if len(text) < size:
        return []
    return [text[index : index + size] for index in range(len(text) - size + 1)]


def meaningful_terms(tokens: list[str]) -> list[str]:
    """Drop generic words so source-name matching focuses on proper terms."""
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "what",
        "which",
        "how",
        "many",
        "rule",
        "rules",
        "ability",
        "abilities",
        "skill",
        "skills",
        "codex",
        "index",
        "中文",
        "规则",
        "技能",
        "种族",
        "是什么",
        "多少",
        "几个",
    }
    return [
        token
        for token in tokens
        if token not in stopwords and (len(token) > 1 or token.isdigit())
    ]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Scale a score vector into the 0..1 range for weighted mixing."""
    if scores.size == 0:
        return scores
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if math.isclose(minimum, maximum):
        fill_value = 1.0 if maximum > 0 else 0.0
        return np.full_like(scores, fill_value, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)
