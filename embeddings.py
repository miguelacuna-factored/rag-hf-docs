"""Factory for the two embedding models compared in this pipeline.

RAG concept: an embedding model turns text into a vector so semantic
similarity becomes a distance calculation — the choice of model changes
that vector space, and therefore what "similar" means for retrieval.
"""

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
}


def get_embedder(embedding_model: str) -> HuggingFaceEmbeddings:
    """Build the `HuggingFaceEmbeddings` instance for `embedding_model` ("minilm" | "bge")."""
    return HuggingFaceEmbeddings(model_name=MODEL_IDS[embedding_model])
