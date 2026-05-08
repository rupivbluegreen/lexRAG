from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import chromadb
import ollama
import structlog
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "eu_regs")
TOP_K = int(os.getenv("TOP_K", "5"))

log = structlog.get_logger("rag")


@dataclass(frozen=True)
class Source:
    text: str
    source: str
    page: int
    score: float


SYSTEM_PROMPT = """You are a careful assistant that answers questions about three EU regulations
(DORA, NIS2, and the AI Act) using only the provided excerpts.

Rules:
- Ground every claim in the excerpts. If the excerpts do not support an answer, say so explicitly.
- Cite sources inline using the format [source p.N], e.g. [DORA.pdf p.42]. Cite after each claim.
- Do not invent article numbers, deadlines, or definitions that are not in the excerpts.
- Prefer concise, structured answers. Use bullet points for lists of obligations or comparisons."""


# Module-level singletons. They survive Streamlit reruns intentionally.
_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_ollama: ollama.Client | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_ollama() -> ollama.Client:
    global _ollama
    if _ollama is None:
        _ollama = ollama.Client(host=OLLAMA_HOST)
    return _ollama


def retrieve(query: str, k: int = TOP_K) -> list[Source]:
    embedder = _get_embedder()
    collection = _get_collection()

    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    res = collection.query(query_embeddings=query_vec, n_results=k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    sources: list[Source] = []
    for text, meta, dist in zip(docs, metas, distances):
        meta = meta or {}
        page_raw = meta.get("page", 0)
        page = int(page_raw) if isinstance(page_raw, (int, float, str)) else 0
        sources.append(
            Source(
                text=text or "",
                source=str(meta.get("source", "unknown")),
                page=page,
                score=float(1.0 - dist),  # cosine distance -> cosine similarity
            )
        )
    return sources


def _format_context(sources: list[Source]) -> str:
    blocks: list[str] = []
    for i, s in enumerate(sources, start=1):
        blocks.append(f"[{i}] ({s.source} p.{s.page})\n{s.text}")
    return "\n\n".join(blocks)


def _build_messages(query: str, sources: list[Source]) -> list[dict[str, str]]:
    context = _format_context(sources)
    user = (
        f"Question: {query}\n\n"
        f"Excerpts:\n{context}\n\n"
        "Answer using only the excerpts above. Cite each claim as [source p.N]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ask(query: str, k: int = TOP_K) -> tuple[str, list[Source]]:
    sources = retrieve(query, k=k)
    messages = _build_messages(query, sources)
    client = _get_ollama()
    resp = client.chat(model=OLLAMA_MODEL, messages=messages, stream=False)
    answer = resp["message"]["content"]
    return answer, sources


def ask_stream(query: str, k: int = TOP_K) -> Iterator[str]:
    sources = retrieve(query, k=k)
    messages = _build_messages(query, sources)
    client = _get_ollama()
    # Yield sources first as a sentinel-free convention: callers wanting them
    # should use `retrieve` or `ask`. ask_stream emits text tokens only.
    for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece
