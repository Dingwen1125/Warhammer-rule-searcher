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
allowed.

## Run

```bash
python3 warhammer_agent.py "星际战士单位可以在推进后冲锋吗？"
python3 warhammer_agent.py "Can a Space Marines unit advance and charge?"
```

## Index New PDFs

After adding PDFs to `knowledge_base/`, precompute embeddings for only new or
changed chunks:

```bash
python3 index_knowledge_base.py
```

The command updates `.cache/warhammer_embeddings.json` and skips chunks that are
already cached.

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
   OCR vision model. OCR output is cached in `.cache/warhammer_ocr.json`.
4. `retriever.py` indexes chunks and performs fast hybrid retrieval:
   embedding similarity plus Chinese/English keyword/IDF scoring.
5. Embeddings are cached in `.cache/warhammer_embeddings.json`, keyed by model,
   document metadata, page, extraction method, and text hash. Existing chunks are
   reused; only new or changed chunks are embedded again.
6. The graph rewrites the question into a concise bilingual rules query,
   retrieves the most relevant chunks across all PDFs, grades relevance, and
   answers only from the cited context.

## Configuration

- `OPENAI_MODEL`: chat/rewrite/relevance/answer model, default `gpt-4.1-mini`
- `OPENAI_OCR_MODEL`: vision OCR model, default same as `OPENAI_MODEL`
- `WARHAMMER_ENABLE_OCR`: set `0` to disable OCR
- `WARHAMMER_OCR_TEXT_MIN_CHARS`: OCR pages with less extracted text than this
  threshold, default `80`
- OCR can be slow on the first run because it calls a vision model for sparse
  pages. Keep `WARHAMMER_ENABLE_OCR=0` for fastest startup if your PDFs already
  contain selectable text.
- `EMBEDDING_MODEL`: configured in `config.py`; changing it automatically creates
  different embedding cache keys
## Project Structure

```text
warhammer_agent.py           CLI entry point and agent application code
index_knowledge_base.py      Precompute and cache missing embeddings
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
