from pathlib import Path

from chunking import chunk_documents
from dataset import load_raw_dataset
from embeddings import get_embedder
from evaluate import build_eval_set, format_results_table, score_retrieval
from ingest import load_documents
from vectorstore import build_collection, collection_name


def build_all_collections(recursive_chunks, semantic_chunks, scope: str) -> None:
    """Embed each chunk set with each embedding model and persist into its own collection."""
    combinations = [
        ("recursive", recursive_chunks, "minilm"),
        ("recursive", recursive_chunks, "bge"),
        ("semantic", semantic_chunks, "minilm"),
        ("semantic", semantic_chunks, "bge"),
    ]
    for chunk_strategy, chunks, embedding_model in combinations:
        store_embedder = get_embedder(embedding_model)
        name = collection_name(chunk_strategy, embedding_model, scope)
        store = build_collection(chunks, store_embedder, name)
        print(f"  [{embedding_model}] '{name}' has {store._collection.count()} vectors")


def build(scope: str) -> None:
    # RAG pipeline:
    # 1 Load raw dataset
    rows = load_raw_dataset()
    # 2. Prepare for ingestion converting to LangChain `Document`s
    docs = load_documents(rows, scope)
    print(f"Loaded {len(docs)} documents")

    # 3. Split into retrievable chunks, once per strategy
    recursive_chunks = chunk_documents(docs, "recursive", scope)
    semantic_chunks = chunk_documents(docs, "semantic", scope, embedder=get_embedder("minilm"))
    print(f"[recursive] chunked into {len(recursive_chunks)} chunks")
    print(f"[semantic] chunked into {len(semantic_chunks)} chunks")

    # 4. Embed and persist all chunk_strategy x embedding_model combinations
    build_all_collections(recursive_chunks, semantic_chunks, scope)


def evaluate(scope: str) -> None:
    rows = load_raw_dataset()
    docs = load_documents(rows, scope)
    eval_set = build_eval_set(docs, scope)
    print(f"Eval set has {len(eval_set)} questions")

    combinations = [
        ("recursive", "minilm"),
        ("recursive", "bge"),
        ("semantic", "minilm"),
        ("semantic", "bge"),
    ]
    results = []
    for chunk_strategy, embedding_model in combinations:
        embedder = get_embedder(embedding_model)
        name = collection_name(chunk_strategy, embedding_model, scope)
        metrics = score_retrieval(eval_set, name, embedder, embedding_model)
        results.append({"chunk_strategy": chunk_strategy, "embedding_model": embedding_model, **metrics})
        print(f"  [{chunk_strategy}/{embedding_model}] {metrics}")

    table = format_results_table(results)
    print(table)
    Path("data/results.md").write_text(table + "\n")


def main():
    build(scope="subset")
    evaluate(scope="subset")


if __name__ == "__main__":
    main()
