"""Generation: turn retrieved chunks + a query into a cited answer.

RAG concept: grounding — the LLM must answer only from the retrieved
context and say so explicitly when it can't, so answers stay traceable to
a source instead of the model's own (possibly wrong) recall.
"""

import os
import re

import anthropic
import logfire
import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from transformers import pipeline

load_dotenv()

LOCAL_MODEL_ID = "google/gemma-2-2b-it"
API_MODEL_ID = "claude-haiku-4-5-20251001"
NOT_IN_CORPUS_PHRASE = "The answer isn't in the corpus."

PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below. Cite sources inline using [1], [2], etc. If the sources don't contain the answer, respond exactly with: "{not_in_corpus}"

{sources}

Question: {query}
Answer:"""


def generate_answer(query: str, chunks: list[Document], backend: str = "local") -> dict:
    """Generate a cited answer to `query` from retrieved `chunks` via the "local" or "api" backend."""
    prompt = _build_prompt(query, chunks)
    if backend == "local":
        raw_answer = _call_local(prompt)
    elif backend == "api":
        raw_answer = _call_api(prompt)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return _parse_answer(raw_answer, chunks)


def _build_prompt(query: str, chunks: list[Document]) -> str:
    sources = "\n\n".join(
        f"[{i}] (source: {chunk.metadata['source']})\n{chunk.page_content}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(not_in_corpus=NOT_IN_CORPUS_PHRASE, sources=sources, query=query)


def _pick_dtype() -> str:
    """Match plan.md's fp16-on-MPS / bf16-on-CUDA split — MPS has spotty bfloat16 support."""
    if torch.cuda.is_available():
        return "bfloat16"
    if torch.backends.mps.is_available():
        return "float16"
    return "float32"


_local_pipeline = None


def _call_local(prompt: str) -> str:
    # No Logfire integration exists for transformers pipelines, so this span is manual —
    # unlike _call_api, it won't get automatic token/cost fields.
    with logfire.span("local_generate", model=LOCAL_MODEL_ID):
        global _local_pipeline
        if _local_pipeline is None:
            _local_pipeline = pipeline(
                "text-generation", model=LOCAL_MODEL_ID, device_map="auto", dtype=_pick_dtype()
            )
        output = _local_pipeline([{"role": "user", "content": prompt}], max_new_tokens=300)
        return output[0]["generated_text"][-1]["content"]


def _call_api(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    response = client.messages.create(
        model=API_MODEL_ID,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_answer(raw_answer: str, chunks: list[Document]) -> dict:
    markers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", raw_answer)))
    citations = [
        {"marker": m, "source": chunks[m - 1].metadata["source"]}
        for m in markers
        if 1 <= m <= len(chunks)
    ]
    not_in_corpus = NOT_IN_CORPUS_PHRASE.lower() in raw_answer.lower()
    grounded = bool(citations) and not not_in_corpus
    return {"answer": raw_answer, "citations": citations, "grounded": grounded}
