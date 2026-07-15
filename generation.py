"""Generation: turn retrieved chunks + a query into a cited answer.

RAG concept: grounding — the LLM must answer only from the retrieved
context and say so explicitly when it can't, so answers stay traceable to
a source instead of the model's own (possibly wrong) recall.
"""

import os
import re
import time

import anthropic
import logfire
import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from transformers import GenerationConfig, pipeline

from langfuse_client import langfuse
from metrics import answers_total, generation_duration_seconds, truncated_total

load_dotenv()

LOCAL_MODEL_ID = "google/gemma-2-2b-it"
API_MODEL_ID = "claude-haiku-4-5-20251001"
NOT_IN_CORPUS_PHRASE = "The answer isn't in the corpus."
# Greedy decoding: given the same retrieved chunks, Gemma should reach the same
# grounded/not-grounded verdict every time, instead of sampling a different one per run.
LOCAL_DO_SAMPLE = False
MAX_NEW_TOKENS = 300

PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below. Cite sources inline using [1], [2], etc. If the sources don't contain the answer, respond exactly with: "{not_in_corpus}"

{sources}

Question: {query}
Answer:"""


def generate_answer(
    query: str, chunks: list[Document], backend: str = LOCAL_MODEL_ID, max_tokens: int = MAX_NEW_TOKENS
) -> dict:
    """Generate a cited answer to `query` from retrieved `chunks` via `backend` (a model ID: LOCAL_MODEL_ID or API_MODEL_ID)."""
    prompt = _build_prompt(query, chunks)
    if backend == LOCAL_MODEL_ID:
        raw_answer = _call_local(prompt, max_tokens)
    elif backend == API_MODEL_ID:
        raw_answer = _call_api(prompt, max_tokens)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    result = _parse_answer(raw_answer, chunks)
    logfire.info(
        "answer parsed",
        backend=backend,
        grounded=result["grounded"],
        num_citations=len(result["citations"]),
        cited_distances=[c["distance"] for c in result["citations"]],
    )
    answers_total.add(1, {"backend": backend, "grounded": result["grounded"]})
    return result


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


def warm_local_pipeline() -> None:
    """Build (once) and cache the local pipeline ahead of a query, so the first `ask()` isn't the cold start."""
    global _local_pipeline
    if _local_pipeline is None:
        _local_pipeline = pipeline("text-generation", model=LOCAL_MODEL_ID, device_map="auto", dtype=_pick_dtype())


def _call_local(prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
    # No Logfire integration exists for transformers pipelines, so token/duration
    # metrics are captured manually here, from the pipeline's own tokenizer and
    # a wall-clock timer — there's no cost field, since local inference has none.
    with (
        logfire.span("local_generate", model=LOCAL_MODEL_ID, max_tokens=max_tokens),
        langfuse.start_as_current_observation(
            as_type="generation", name="local_generate", model=LOCAL_MODEL_ID, input=prompt
        ) as gen,
    ):
        warm_local_pipeline()
        # A GenerationConfig built from scratch (rather than max_new_tokens= as a bare
        # kwarg) avoids colliding with the model's inherited default max_length=20 —
        # see https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        generation_config = GenerationConfig(max_new_tokens=max_tokens, do_sample=LOCAL_DO_SAMPLE)
        start = time.perf_counter()
        output = _local_pipeline([{"role": "user", "content": prompt}], generation_config=generation_config)
        duration_seconds = time.perf_counter() - start
        answer = output[0]["generated_text"][-1]["content"]
        tokenizer = _local_pipeline.tokenizer
        input_tokens = len(tokenizer(prompt)["input_ids"])
        output_tokens = len(tokenizer(answer)["input_ids"])
        # No explicit stop-reason from transformers, so truncation is inferred:
        # hitting the cap exactly means it almost certainly didn't reach EOS naturally.
        truncated = output_tokens >= max_tokens
        logfire.info(
            "local generation complete",
            backend=LOCAL_MODEL_ID,
            duration_seconds=duration_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            truncated=truncated,
        )
        gen.update(
            output=answer,
            usage_details={"input": input_tokens, "output": output_tokens},
            metadata={"duration_seconds": duration_seconds, "truncated": truncated},
        )
        generation_duration_seconds.record(duration_seconds, {"backend": LOCAL_MODEL_ID})
        if truncated:
            truncated_total.add(1, {"backend": LOCAL_MODEL_ID})
        return answer


def _call_api(prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    with langfuse.start_as_current_observation(
        as_type="generation", name="api_generate", model=API_MODEL_ID, input=prompt
    ) as gen:
        start = time.perf_counter()
        response = client.messages.create(
            model=API_MODEL_ID,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        duration_seconds = time.perf_counter() - start
        truncated = response.stop_reason == "max_tokens"
        logfire.info(
            "api generation complete",
            backend=API_MODEL_ID,
            duration_seconds=duration_seconds,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            truncated=truncated,
        )
        gen.update(
            output=response.content[0].text,
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
            metadata={"duration_seconds": duration_seconds, "truncated": truncated, "stop_reason": response.stop_reason},
        )
        generation_duration_seconds.record(duration_seconds, {"backend": API_MODEL_ID})
        if truncated:
            truncated_total.add(1, {"backend": API_MODEL_ID})
        return response.content[0].text


def _parse_answer(raw_answer: str, chunks: list[Document]) -> dict:
    markers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", raw_answer)))
    citations = [
        {"marker": m, "source": chunks[m - 1].metadata["source"], "distance": chunks[m - 1].metadata["distance"]}
        for m in markers
        if 1 <= m <= len(chunks)
    ]
    not_in_corpus = NOT_IN_CORPUS_PHRASE.lower() in raw_answer.lower()
    grounded = bool(citations) and not not_in_corpus
    return {"answer": raw_answer, "citations": citations, "grounded": grounded}
