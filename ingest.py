"""Ingestion: turn raw dataset rows into LangChain `Document`s.

RAG concept: every downstream stage (chunking, embedding, retrieval) needs a
consistent unit to work with, and citations need a stable identifier that
survives all the way from "raw row" to "chunk shown to the user." This is the
one place raw rows get shaped into that unit, so the citation trail starts
here instead of being reconstructed later.
"""

from datasets import Dataset
from langchain_core.documents import Document

SUBSET_SIZE = 200 # TODO: changet to full once we are done with the pipeline


def load_documents(rows: Dataset, scope: str = "subset") -> list[Document]:
    """Convert raw dataset rows into `Document`s keyed by their source path."""
    if scope == "subset":
        rows = rows.select(range(min(SUBSET_SIZE, len(rows))))

    return [
        Document(page_content=row["text"], metadata={"source": row["source"]})
        for row in rows
    ]
