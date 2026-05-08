# CLAUDE.md

Operational guide for AI coding agents working on this repo. Read this first; the README is for humans evaluating the project.

## What this is

A local-only RAG prototype: chat with three EU regulations (DORA, NIS2, AI Act) using Ollama + BGE-small + ChromaDB. Three Python modules, ~370 LOC total. **V1 is intentionally naive.** Do not "fix" naive choices unless the task explicitly asks for V2 work — the V1 weaknesses are evidence for an upcoming write-up.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull llama3.1:8b          # one-time
./scripts/download_corpus.sh     # one-time, fetches 3 PDFs from EUR-Lex
python ingest.py                 # builds the vector store

streamlit run app.py             # the chat UI
```

Ollama must be running on `http://localhost:11434` (the macOS app or `ollama serve`).

## Repo map

```
ingest.py            parse → chunk → embed → index (one-shot CLI)
rag.py               retrieve + generate (pure functions, no UI)
app.py               Streamlit chat (calls rag.py only)
scripts/             corpus download
pdfs/                corpus PDFs (gitignored)
data/chroma/         persistent vector store (gitignored)
.env                 local config (gitignored)
```

## Module contracts

**`ingest.py`** — CLI entry point. Drops and recreates the `eu_regs` collection on every run. If you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBED_MODEL` in `.env`, you must re-run `python ingest.py`.

**`rag.py`** — public API:
- `retrieve(query, k) -> list[Source]`
- `ask(query, k) -> tuple[str, list[Source]]` (non-streaming)
- `ask_stream(query, k) -> Iterator[str]` (streaming tokens)

Uses module-level singletons for the embedder, Chroma collection, and Ollama client. They survive Streamlit reruns intentionally — do not move them inside functions.

**`app.py`** — Streamlit only. Imports from `rag.py`. Should never call ChromaDB, Ollama, or `sentence-transformers` directly.

## Configuration

All config flows through `.env` → `os.getenv` at module load. Defaults live in `.env.example`. Do not hardcode model names, paths, or chunk sizes anywhere else.

| Variable          | Default                       | Notes                                 |
| ----------------- | ----------------------------- | ------------------------------------- |
| `OLLAMA_HOST`     | `http://localhost:11434`      |                                       |
| `OLLAMA_MODEL`    | `llama3.1:8b`                 | Any Ollama-pulled model works         |
| `EMBED_MODEL`     | `BAAI/bge-small-en-v1.5`      | Changing requires re-ingest           |
| `CHUNK_SIZE`      | `1000`                        | chars; changing requires re-ingest    |
| `CHUNK_OVERLAP`   | `200`                         | chars; changing requires re-ingest    |
| `TOP_K`           | `5`                           | runtime, no re-ingest needed          |
| `CHROMA_DIR`      | `./data/chroma`               |                                       |
| `COLLECTION_NAME` | `eu_regs`                     |                                       |
| `PDF_DIR`         | `./pdfs`                      |                                       |

## Conventions

- **Logging**: `structlog` for any new module-level logging. No `print()` in library code.
- **Typing**: Type hints everywhere. `from __future__ import annotations` at the top of every module.
- **Imports**: stdlib → third-party → local, alphabetised within each block.
- **Style**: Plain Python. No classes for the sake of classes; prefer pure functions and small dataclasses. The existing `Source` dataclass is the pattern.
- **No agent frameworks.** Do not import `langchain` (other than `langchain-text-splitters`), `llama-index`, `langgraph`, or `crewai`. If you think one is needed, stop and ask.
- **No new heavyweight deps without checking.** Anything > 50 MB on disk or with native build steps requires confirmation.

## Invariants — do not break

1. **Local-only is non-negotiable.** No outbound calls to OpenAI, Anthropic, Cohere, Voyage, or any hosted API. Every model and store runs on the user's machine.
2. **`rag.py` has no UI imports.** It must remain reusable from a future FastAPI service in V2.
3. **The Chroma collection is single-source-of-truth at runtime.** Don't add a second cache layer in V1.
4. **Citations must round-trip.** Every chunk in the store has `source` (filename) and `page` (1-indexed) metadata. Don't add a code path that loses them.
5. **The corpus is public-domain regulatory text.** No code path may write user queries or model outputs to disk or telemetry — V2 will introduce structured tracing under explicit consent.

## V1 vs V2 — scope discipline

V1 is **complete and shippable as-is**. The known weaknesses are documented in the README and are deliberate.

V2 work, in priority order, is **out of scope unless the task explicitly references V2**:

1. Hybrid retrieval (BM25 + dense) via `rank_bm25`
2. Reranker (`bge-reranker-v2-m3`, local)
3. Semantic chunking
4. Inline citations with span highlighting
5. Eval harness (faithfulness, answer relevance, context precision)
6. OpenTelemetry tracing — retrieval / rerank / generation latency, token cost
7. FastAPI service layer behind Streamlit
8. Query rewriting / HyDE

If the user asks for a V1 bug fix, fix only that bug. Do not opportunistically add V2 features.

## How to validate a change

There is no automated test suite in V1. After any change:

1. `python -c "import ingest, rag, app"` — imports clean.
2. `python ingest.py` — completes without error and reports a non-zero chunk count for all three PDFs.
3. `streamlit run app.py` — UI loads, accepts a query, returns an answer with at least one cited source.
4. Try the three canonical questions:
   - *"What does DORA require for ICT third-party risk management?"*
   - *"Compare NIS2 incident reporting deadlines with DORA's."*
   - *"What are the prohibited AI practices under the AI Act?"*

A V2 task will additionally require the new eval harness to pass; until then, manual smoke-testing is the contract.

## Agent etiquette

- **Show before doing.** For any change touching more than one file or more than ~30 lines, propose the diff before applying.
- **Surgical edits over rewrites.** Match the existing style; do not reformat unrelated code.
- **No silent dependency additions.** If you add anything to `requirements.txt`, call it out in your reply with the reason.
- **Commit messages**: imperative mood, scoped (`ingest:`, `rag:`, `app:`, `scripts:`, `docs:`).
- **When unsure, ask.** This codebase is small enough that the cost of a clarifying question is lower than the cost of a wrong refactor.
