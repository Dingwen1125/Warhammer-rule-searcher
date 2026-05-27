from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL
from state import RagState


def agent(state: RagState) -> RagState:
    route_text = f"{state['question']} {state.get('search_query', '')}".lower()
    warhammer_terms = {
        "warhammer",
        "40k",
        "40,000",
        "age of sigmar",
        "aos",
        "codex",
        "index",
        "datasheet",
        "detachment",
        "stratagem",
        "army rule",
        "battle round",
        "command phase",
        "movement phase",
        "shooting phase",
        "charge phase",
        "fight phase",
        "objective control",
        "oc",
        "save",
        "invulnerable",
        "feel no pain",
        "leader",
        "aura",
        "unit",
        "model",
        "weapon",
        "wound",
        "mortal wound",
        "faction",
        "space marines",
        "tyranids",
        "orks",
        "necrons",
        "chaos",
        "规则",
        "战锤",
        "单位",
        "阵营",
        "武器",
        "阶段",
        "冲锋",
        "射击",
    }
    if any(term in route_text for term in warhammer_terms):
        return {"should_retrieve": True}

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    prompt = f"""
You are a routing agent for a Warhammer rules RAG workflow.
Decide whether answering this question requires retrieving from the local rule PDF knowledge base.
Return only RETRIEVE or END.

Original question: {state["question"]}
Rules search query: {state.get("search_query", "")}
"""
    decision = str(llm.invoke(prompt).content).strip().upper()
    if "END" in decision and "RETRIEVE" not in decision:
        return {
            "should_retrieve": False,
            "answer": (
                "This workflow is configured to answer Warhammer rules questions "
                "from the local PDF knowledge base."
            ),
        }
    return {"should_retrieve": True}


def should_retrieve(state: RagState) -> Literal["continue", "end"]:
    return "continue" if state.get("should_retrieve", True) else "end"
