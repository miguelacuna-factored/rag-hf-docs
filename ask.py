"""ask.py — the RAG step itself: retrieval + generation combined.

RAG concept: this is Retrieval-Augmented Generation in one call — retrieve
the chunks most relevant to a query, then let the LLM generate an answer
grounded in only those chunks, instead of its own unconstrained recall.
"""

import logfire
from langchain_huggingface import HuggingFaceEmbeddings

from generation import LOCAL_MODEL_ID, generate_answer
from retrieval import search


def ask(
    query: str,
    collection_name: str,
    embedder: HuggingFaceEmbeddings,
    embedding_model: str,
    backend: str = LOCAL_MODEL_ID,
) -> dict:
    """Run retrieval then generation for `query`; skips generation if nothing relevant was found."""
    with logfire.span(
        "ask", query=query, collection_name=collection_name, embedding_model=embedding_model, backend=backend
    ):
        chunks = search(query, collection_name, embedder, embedding_model)
        if not chunks:
            return {"answer": "No relevant information found in the corpus.", "citations": [], "grounded": False}
        return generate_answer(query, chunks, backend)
