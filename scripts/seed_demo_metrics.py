"""seed_demo_metrics.py — fire a batch of varied queries to populate Logfire with real
metric data (queries_total, answers_total, generation_duration_seconds, truncated_total,
errors_total) ahead of a dashboard/alert demo.

Run from the repo root: uv run python scripts/seed_demo_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import run_query
from generation import API_MODEL_ID, LOCAL_MODEL_ID
from vectorstore import list_built_scopes

SCOPE = "full" if "full" in list_built_scopes() else "subset"

# Each case: (chunk_strategy, embedding_model, backend, query, chunks_top_k, max_tokens)
CASES = [
    ("recursive", "minilm", LOCAL_MODEL_ID, "what is RAG?", 5, 300),
    ("semantic", "bge", LOCAL_MODEL_ID, "what is a tokenizer?", 5, 300),
    ("recursive", "minilm", API_MODEL_ID, "what is fine-tuning?", 5, 300),
    ("semantic", "bge", API_MODEL_ID, "how does attention work?", 5, 300),
    # Deliberately truncated: a low max_tokens cap on a question that needs a longer answer.
    ("recursive", "minilm", LOCAL_MODEL_ID, "explain the transformer architecture in detail", 5, 50),
    ("recursive", "minilm", API_MODEL_ID, "explain the transformer architecture in detail", 5, 50),
    # Deliberately ungrounded: not answerable from the Hugging Face docs corpus.
    ("recursive", "minilm", LOCAL_MODEL_ID, "what is the capital of France?", 5, 300),
]

# Nice-to-have: an intentionally broken call, to populate errors_total for the alert demo.
# Not critical if this fails differently than expected — it's wrapped so it never aborts the run.
ERROR_CASE = ("recursive", "minilm", "bogus-backend-id", "what is RAG?", 5, 300)


def run_case(chunk_strategy, embedding_model, backend, query, chunks_top_k, max_tokens):
    history, _ = run_query(query, chunk_strategy, embedding_model, backend, SCOPE, chunks_top_k, max_tokens, [])
    entry = history[-1]
    grounded = "grounded" if entry["grounded"] else "ungrounded"
    print(f"[{backend} | {chunk_strategy} | {embedding_model}] {grounded}: {entry['answer'][:80]!r}")


if __name__ == "__main__":
    print(f"Seeding demo metrics against scope={SCOPE!r} ({len(CASES)} cases + 1 nice-to-have error case)\n")
    for case in CASES:
        run_case(*case)

    try:
        run_case(*ERROR_CASE)
    except Exception as e:
        print(f"[error case] raised outside run_query's own handling ({e!r}) — still fine, this is a nice-to-have.")

    print("\nDone. Check the Logfire dashboard/live view for the new data.")
