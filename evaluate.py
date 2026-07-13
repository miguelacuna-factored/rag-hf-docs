"""Evaluation: measure retrieval quality across chunk_strategy x embedding_model.

RAG concept: chunking and embedding choices are usually invisible until you
measure them against real questions — this stage turns "which combination
retrieves better" into numbers instead of a guess.
"""

import json
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from cache import load_cached_or_build
from retrieval import search

load_dotenv()

DATA_DIR = Path("data")
EVAL_SET_SIZE = 25
API_MODEL_ID = "claude-haiku-4-5-20251001"
RANDOM_SEED = 42

QUESTION_PROMPT = """The following is an excerpt from a Hugging Face documentation page. Write ONE natural question that this excerpt directly answers. Respond with only the question, nothing else.

{excerpt}"""


def build_eval_set(docs: list[Document], scope: str) -> list[dict]:
    """Sample docs and generate one question per doc via Claude, cached under `./data/eval_set_{scope}.json`."""
    cache_path = DATA_DIR / f"eval_set_{scope}.json"
    return load_cached_or_build(
        cache_path,
        build_fn=lambda: _build_eval_set(docs),
        loader=lambda p: json.loads(p.read_text()),
        saver=lambda eval_set, p: p.write_text(json.dumps(eval_set, indent=2)),
    )


def _build_eval_set(docs: list[Document]) -> list[dict]:
    random.seed(RANDOM_SEED)
    sampled = random.sample(docs, min(EVAL_SET_SIZE, len(docs)))
    return [
        {"query": question, "source": doc.metadata["source"]}
        for doc in sampled
        if (question := _generate_question(doc))
    ]


def _generate_question(doc: Document) -> str:
    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    response = client.messages.create(
        model=API_MODEL_ID,
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION_PROMPT.format(excerpt=doc.page_content[:2000])}],
    )
    return response.content[0].text.strip()


def score_retrieval(
    eval_set: list[dict], collection_name: str, embedder: HuggingFaceEmbeddings, embedding_model: str, k: int = 5
) -> dict:
    """Compute Hit Rate@k, MRR@k, Precision@k for one collection over `eval_set`."""
    hits, reciprocal_ranks, precisions = [], [], []
    for item in eval_set:
        results = search(item["query"], collection_name, embedder, embedding_model, k=k)
        correct_ranks = [rank for rank, doc in enumerate(results, start=1) if doc.metadata["source"] == item["source"]]
        hits.append(1 if correct_ranks else 0)
        reciprocal_ranks.append(1 / correct_ranks[0] if correct_ranks else 0)
        precisions.append(len(correct_ranks) / k)
    n = len(eval_set)
    return {"hit_rate": sum(hits) / n, "mrr": sum(reciprocal_ranks) / n, "precision": sum(precisions) / n}


def format_results_table(rows: list[dict]) -> str:
    """Render `rows` (each with chunk_strategy, embedding_model, hit_rate, mrr, precision) as a markdown table."""
    header = "| chunk_strategy | embedding_model | hit_rate@k | mrr@k | precision@k |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['chunk_strategy']} | {r['embedding_model']} | {r['hit_rate']:.2f} | {r['mrr']:.2f} | {r['precision']:.2f} |"
        )
    return "\n".join(lines)
