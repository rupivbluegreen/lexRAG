# lexRAG

A small, deliberately simple, **local-only** RAG over three EU regulations:

- **DORA** — Regulation (EU) 2022/2554, digital operational resilience for the financial sector
- **NIS2** — Directive (EU) 2022/2555, common cybersecurity baseline
- **AI Act** — Regulation (EU) 2024/1689, harmonised rules on AI

Everything runs on the user's machine: PDFs from EUR-Lex, BGE-small embeddings via `sentence-transformers`, persistent ChromaDB store, generation through Ollama. No hosted APIs.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull llama3.1:8b
./scripts/download_corpus.sh
python ingest.py

streamlit run app.py
```

## Architecture

Three modules, ~370 LOC:

- `ingest.py` — pypdf parse → per-page chunking with `RecursiveCharacterTextSplitter` → BGE-small embeddings → drop-and-recreate Chroma collection.
- `rag.py` — `retrieve` (k nearest chunks), `ask` (one-shot), `ask_stream` (token stream). Module-level singletons for the embedder, Chroma collection, and Ollama client.
- `app.py` — Streamlit chat. Imports only from `rag.py`.

Every chunk carries `source` (PDF filename) and `page` (1-indexed) so citations round-trip cleanly.

## Known V1 weaknesses (deliberate)

These are the targets for V2:

- **Dense-only retrieval.** No BM25, no hybrid scoring. Lexical matches on article numbers and defined terms are weaker than they should be.
- **No reranker.** Top-K from cosine similarity is taken at face value.
- **Naive chunking.** Pure character-based recursive splitting, indifferent to article and recital boundaries.
- **No eval harness.** Quality is judged by smoke-testing three canonical questions.
- **No tracing.** Latency and token cost are not measured.
- **Streamlit is the API.** No FastAPI layer underneath.
- **No query rewriting.** User queries hit the embedder verbatim.

## License & corpus notes

The PDFs in `pdfs/` are public-domain regulatory text from EUR-Lex and are not redistributed in this repo (`.gitignore`d). Run `./scripts/download_corpus.sh` to fetch them.
