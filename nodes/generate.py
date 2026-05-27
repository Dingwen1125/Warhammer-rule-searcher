"""Answer-generation node grounded only in retrieved PDF context."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from language import contains_chinese
from retriever import format_context
from state import RagState


def generate(state: RagState) -> RagState:
    """Generate a final answer from relevant retrieved chunks only."""
    if not state.get("relevant_chunks"):
        return {
            "answer": (
                "I could not find relevant information in the PDF knowledge base, "
                "so I will not answer this question."
            )
        }

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chunks = state["relevant_chunks"]
    context = format_context(chunks)
    prompt = f"""
You are a careful Warhammer rules assistant. Answer only from the provided PDF
context. If the context does not contain enough information, say what is missing.
When relevant chunks come from different files, identify which source you are
using and do not merge incompatible rules from different game systems, editions,
factions, or codexes. Cite source file and page numbers in the answer.
If the question asks about a faction, army, codex, unit, or rules source that is
not present in the PDF context, say that the knowledge base does not contain that
source instead of answering from a different faction or codex.
The user's question language is {state.get("answer_language", "English")}.
You must answer in {state.get("answer_language", "English")}. Do not switch to
the PDF context language when it differs from the user's question language.

Question: {state["question"]}

PDF context:
{context}
"""
    answer = str(llm.invoke(prompt).content).strip()
    answer = enforce_answer_language(answer, state.get("answer_language", "English"), llm)
    return {"answer": answer}


def enforce_answer_language(answer: str, answer_language: str, llm: ChatOpenAI) -> str:
    """Translate the answer if the model ignored the requested output language."""
    if answer_language == "English" and is_mostly_chinese(answer):
        return translate_answer(answer, "English", llm)
    if answer_language == "Chinese" and not contains_chinese(answer):
        return translate_answer(answer, "Chinese", llm)
    return answer


def is_mostly_chinese(text: str) -> bool:
    """Detect whether the answer body is predominantly Chinese text."""
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    return chinese_chars > max(40, latin_chars)


def translate_answer(answer: str, target_language: str, llm: ChatOpenAI) -> str:
    """Translate an already-grounded answer while preserving citations and terms."""
    prompt = f"""
Translate the following Warhammer rules answer into {target_language}. Preserve
source citations, page numbers, unit names, rule names, and PDF file names. Do
not add new information or change the meaning.

Answer:
{answer}
"""
    return str(llm.invoke(prompt).content).strip()
