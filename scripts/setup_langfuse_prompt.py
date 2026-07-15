"""setup_langfuse_prompt.py — one-time upload of the RAG answer prompt into Langfuse
Prompt Management, so it can be edited from the Langfuse UI without a redeploy.

`get_prompt(PROMPT_NAME)` (see generation.py) defaults to fetching the "production"-labeled
version, so this script creates the first version under that label.

Run from the repo root: uv run python scripts/setup_langfuse_prompt.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation import FALLBACK_PROMPT_TEMPLATE, PROMPT_NAME
from langfuse_client import langfuse

if __name__ == "__main__":
    prompt = langfuse.create_prompt(
        name=PROMPT_NAME,
        prompt=FALLBACK_PROMPT_TEMPLATE,
        labels=["production"],
        commit_message="Initial version, migrated from generation.py's hardcoded template",
    )
    langfuse.flush()
    print(f"Created {PROMPT_NAME!r} version {prompt.version} with label 'production'.")
    print("Edit it from the Langfuse UI under Prompts — the app picks up changes within 5 minutes (cache_ttl_seconds=300).")
