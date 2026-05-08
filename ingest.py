from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import chromadb
import structlog
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "eu_regs")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")

EMBED_BATCH = 64

log = structlog.get_logger("ingest")


def iter_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, text), ...] for a PDF, dropping empty pages."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf occasionally trips on malformed pages
            log.warning("page_extract_failed", file=pdf_path.name, page=idx, error=str(exc))
            continue
        text = text.strip()
        if text:
            pages.append((idx, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[int, str]]:
    """Split each page independently so every chunk maps to exactly one page."""
    chunks: list[tuple[int, str]] = []
    for page_no, page_text in pages:
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if piece:
                chunks.append((page_no, piece))
    return chunks


def main() -> int:
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.is_dir():
        log.error("pdf_dir_missing", path=str(pdf_dir))
        return 1

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.error("no_pdfs_found", path=str(pdf_dir))
        return 1

    log.info(
        "config",
        embed_model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chroma_dir=CHROMA_DIR,
        collection=COLLECTION_NAME,
        pdfs=[p.name for p in pdf_files],
    )

    embedder = SentenceTransformer(EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("collection_dropped", name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for pdf in pdf_files:
        pages = iter_pages(pdf)
        chunks = chunk_pages(pages, splitter)
        if not chunks:
            log.warning("no_chunks", file=pdf.name)
            continue

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, str | int | float | bool]] = []
        for i, (page_no, text) in enumerate(chunks):
            ids.append(f"{pdf.stem}-p{page_no}-c{i}")
            docs.append(text)
            metas.append({"source": pdf.name, "page": page_no})

        for start in range(0, len(docs), EMBED_BATCH):
            end = start + EMBED_BATCH
            embeddings = embedder.encode(
                docs[start:end],
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()
            collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=cast(Any, metas[start:end]),
                embeddings=embeddings,
            )

        total += len(chunks)
        log.info("indexed", file=pdf.name, pages=len(pages), chunks=len(chunks))

    log.info("done", total_chunks=total, collection=COLLECTION_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
