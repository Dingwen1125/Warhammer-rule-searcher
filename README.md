# Warhammer Rules LangGraph RAG Agent

This project builds a custom RAG agent for Chinese and English Warhammer rule
PDFs. It can read multiple local PDFs, OCR image-heavy pages with a vision
model, use fast hybrid keyword/vector retrieval to select the right
rulebook/codex/faction file, and answer with source/page citations.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Put one or more Warhammer rule PDFs under `knowledge_base/`. Chinese, English,
and mixed-language PDFs are supported. Nested folders are
allowed, for example:

```text
knowledge_base/
  核心规则_第10版.pdf
  星际战士圣典.pdf
  core_rules_10th.pdf
  codex_space_marines.pdf
  泰伦虫族圣典.pdf
  西格玛时代/雷铸神兵.pdf
```

## Run

```bash
python3 warhammer_agent.py "星际战士单位可以在推进后冲锋吗？"
python3 warhammer_agent.py "Can a Space Marines unit advance and charge?"
```

## Workflow

```text
Prepare Query -> Rewrite -> Agent -> Should Retrieve -> Tool -> Check Relevance -> Generate -> Answer
                                      |
                                      No
                                      v
                                   Rewrite -> Agent
```

1. `knowledge_base.py` finds every PDF under `knowledge_base/`.
2. Text pages are extracted with `pypdf`.
3. Sparse/image-heavy pages are sent through `nodes/process_images.py`, which
   renders the page with PyMuPDF and extracts visible words with the configured
   OCR vision model.
4. `retriever.py` indexes chunks and performs fast hybrid retrieval:
   embedding similarity plus Chinese/English keyword/IDF scoring.
5. The graph rewrites the question into a concise bilingual rules query,
   retrieves the most relevant chunks across all PDFs, grades relevance, and
   answers only from the cited context.

## Configuration

- `OPENAI_MODEL`: chat/rewrite/relevance/answer model, default `gpt-4.1-mini`
- `OPENAI_OCR_MODEL`: vision OCR model, default same as `OPENAI_MODEL`
- `WARHAMMER_ENABLE_OCR`: set `0` to disable OCR
- `WARHAMMER_OCR_TEXT_MIN_CHARS`: OCR pages with less extracted text than this
  threshold, default `80`
## Project Structure

```text
warhammer_agent.py           CLI entry point and agent application code
graph.py                    LangGraph workflow assembly
state.py                    Shared RagState and Chunk types
config.py                   Model, OCR, and knowledge-base settings
env.py                      Local .env loader
knowledge_base.py           Multi-PDF extraction, OCR, and chunking
retriever.py                Fast hybrid retrieval and context formatting
nodes/agent.py              Agent node and should-retrieve route
nodes/process_images.py     OCR node for image-heavy PDF pages
nodes/prepare_query.py      Detects answer language
nodes/retrieve_tool.py      Retrieval tool node
nodes/check_relevance.py    Relevance grading node and route
nodes/rewrite_query.py      Query rewrite node
nodes/generate.py           Answer generation node
nodes/no_answer.py          Stops when PDF context is not relevant
```
