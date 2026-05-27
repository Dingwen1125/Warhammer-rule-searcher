from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from retriever import format_context
from state import RagState


def generate(state: RagState) -> RagState:
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
Answer in {state.get("answer_language", "English")}, matching the user's question
language even if the PDF context is in another language.

Question: {state["question"]}

PDF context:
{context}
"""
    answer = llm.invoke(prompt).content
    return {"answer": str(answer).strip()}
