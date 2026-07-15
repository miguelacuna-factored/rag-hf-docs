"""ask.py — the RAG step itself: retrieval + generation combined.

RAG concept: this is Retrieval-Augmented Generation in one call — retrieve
the chunks most relevant to a query, then let the LLM generate an answer
grounded in only those chunks, instead of its own unconstrained recall.
"""

import logfire
from langchain_huggingface import HuggingFaceEmbeddings

from generation import LOCAL_MODEL_ID, MAX_NEW_TOKENS, generate_answer
from metrics import queries_total
from retrieval import DEFAULT_CHUNKS_TOP_K, search


def ask(
    query: str,
    collection_name: str,
    embedder: HuggingFaceEmbeddings,
    embedding_model: str,
    backend: str = LOCAL_MODEL_ID,
    chunks_top_k: int = DEFAULT_CHUNKS_TOP_K,
    max_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run retrieval then generation for `query`; skips generation if nothing relevant was found."""
    with logfire.span(
        "ask",
        query=query,
        collection_name=collection_name,
        embedding_model=embedding_model,
        backend=backend,
        chunks_top_k=chunks_top_k,
        max_tokens=max_tokens,
    ):
        queries_total.add(1, {"backend": backend, "embedding_model": embedding_model, "collection_name": collection_name})
        chunks = search(query, collection_name, embedder, embedding_model, chunks_top_k=chunks_top_k)
        if not chunks:
            return {"answer": "No relevant information found in the corpus.", "citations": [], "grounded": False}
        return generate_answer(query, chunks, backend, max_tokens)
