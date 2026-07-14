"""Retrieval: pull relevant chunks back out for a query.

RAG concept: retrieval turns a query into the same vector space the corpus
was embedded into, then finds nearest neighbors — this is the step that
actually determines what context the LLM gets to answer from.
"""

import logfire
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore import VECTORDB_DIR

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(query: str, name: str, embedder: HuggingFaceEmbeddings, embedding_model: str, k: int = 5) -> list[Document]:
    """Semantic search over Chroma collection `name`; applies the BGE query-instruction prefix when embedding_model == "bge"."""
    with logfire.span("retrieval", name=name, embedding_model=embedding_model, k=k):
        store = Chroma(collection_name=name, embedding_function=embedder, persist_directory=str(VECTORDB_DIR))
        search_query = f"{BGE_QUERY_PREFIX}{query}" if embedding_model == "bge" else query
        scored = store.similarity_search_with_score(search_query, k=k)
        chunks = [
            {"source": doc.metadata["source"], "distance": score, "length": len(doc.page_content)}
            for doc, score in scored
        ]
        distances = [c["distance"] for c in chunks]
        logfire.info(
            "retrieved {count} chunks",
            count=len(chunks),
            chunks=chunks,
            min_distance=min(distances, default=None),
            max_distance=max(distances, default=None),
        )
        return [doc for doc, _ in scored]
