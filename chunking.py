"""Chunking: split `Document`s into retrievable pieces.

RAG concept: an embedding model can only represent a chunk of text as one
vector, so chunk boundaries decide what gets compared against a query at
retrieval time — too big and irrelevant text dilutes the match, too small
and necessary context gets split across chunks the retriever won't join
back together.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cache import load_cached_or_build

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DATA_DIR = Path("data")


def chunk_documents(
    docs: list[Document],
    chunk_strategy: str,
    scope: str,
    embedder: HuggingFaceEmbeddings | None = None,
) -> list[Document]:
    """Split `docs` per `chunk_strategy` ("recursive" | "semantic"), cached under `./data/`."""
    cache_path = DATA_DIR / f"chunks_{chunk_strategy}_{scope}.json"
    return load_cached_or_build(
        cache_path,
        build_fn=lambda: _build_chunks(docs, chunk_strategy, scope, embedder),
        loader=_load_chunks,
        saver=_save_chunks,
    )


def _build_chunks(
    docs: list[Document], chunk_strategy: str, scope: str, embedder: HuggingFaceEmbeddings | None
) -> list[Document]:
    if chunk_strategy == "recursive":
        chunks = _chunk_recursive(docs)
    elif chunk_strategy == "semantic":
        chunks = _chunk_semantic(docs, embedder)
    else:
        raise ValueError(f"Unknown chunk_strategy: {chunk_strategy!r}")

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = chunk_strategy
        chunk.metadata["scope"] = scope
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


def _chunk_recursive(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def _chunk_semantic(docs: list[Document], embedder: HuggingFaceEmbeddings | None) -> list[Document]:
    splitter = SemanticChunker(embedder)
    return splitter.split_documents(docs)


def _load_chunks(path: Path) -> list[Document]:
    rows = json.loads(path.read_text())
    return [Document(page_content=row["page_content"], metadata=row["metadata"]) for row in rows]


def _save_chunks(chunks: list[Document], path: Path) -> None:
    rows = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    path.write_text(json.dumps(rows))
