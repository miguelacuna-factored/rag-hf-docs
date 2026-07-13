"""Vector store: build/load the Chroma collections chunks get embedded into.

RAG concept: embedding every chunk is expensive, but the resulting vectors
don't change unless the chunks or embedding model do — so this stage
persists them once to disk, letting every later query reuse that work
instead of re-embedding the whole corpus per search.
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VECTORDB_DIR = Path("vectordb")


def collection_name(chunk_strategy: str, embedding_model: str, scope: str) -> str:
    """Build the canonical Chroma collection name for a (chunk_strategy, embedding_model, scope) combo."""
    return f"{chunk_strategy}_{embedding_model}_{scope}"


def list_built_scopes() -> list[str]:
    """Which of "subset"/"full" have at least one collection already built."""
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    built = {c.name.rsplit("_", 1)[-1] for c in client.list_collections()}
    return [s for s in ["subset", "full"] if s in built]


def build_collection(chunks: list[Document], embedder: HuggingFaceEmbeddings, name: str) -> Chroma:
    """Embed and persist `chunks` into Chroma collection `name`, skipping rebuild if already populated."""
    store = Chroma(collection_name=name, embedding_function=embedder, persist_directory=str(VECTORDB_DIR))
    if store._collection.count() == 0:
        # Chroma rejects a single upsert bigger than its max batch size, so
        # large chunk sets (e.g. the "full" scope) must go in over several calls.
        batch_size = store._client.get_max_batch_size()
        for i in range(0, len(chunks), batch_size):
            store.add_documents(chunks[i : i + batch_size])
    return store
