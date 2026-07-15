"""app.py — Gradio UI for comparing chunk_strategy x embedding_model x backend combinations via a history log."""

from dotenv import load_dotenv

load_dotenv()

import logfire

# rag-hf-docs sends telemetry to its own Logfire project (us region); LOGFIRE_TOKEN in .env picks the project.
logfire.configure(
    service_name="rag-hf-docs",
    service_version="0.1.0",
    environment="local",
    advanced=logfire.AdvancedOptions(base_url="https://logfire-us.pydantic.dev"),
)
logfire.instrument_anthropic()  # covers the "api" backend; the local HF backend gets a manual span (see generation.py)

from datetime import datetime

import gradio as gr
from langfuse import propagate_attributes

from ask import ask
from embeddings import get_embedder
from generation import LOCAL_MODEL_ID, MAX_NEW_TOKENS, warm_local_pipeline
from langfuse_client import langfuse
from metrics import errors_total
from retrieval import DEFAULT_CHUNKS_TOP_K
from vectorstore import collection_name, list_built_scopes

CHUNK_STRATEGIES = ["recursive", "semantic"]
EMBEDDING_MODELS = ["minilm", "bge"]
BACKENDS = ["google/gemma-2-2b-it", "claude-haiku-4-5-20251001"]


def warm_embedder(embedding_model: str) -> None:
    """Build (once) and cache the embedder for `embedding_model` as soon as it's selected, ahead of "Ask"."""
    get_embedder(embedding_model)


def warm_backend(backend: str) -> None:
    """Build (once) and cache the local pipeline as soon as it's selected; the api backend has nothing to warm."""
    if backend == LOCAL_MODEL_ID:
        warm_local_pipeline()


def format_entry(entry: dict) -> str:
    citations = (
        "\n".join(f"- ({c['distance']:.4f}) [{c['marker']}] {c['source']}" for c in entry["citations"]) or "none"
    )
    grounded = "✅" if entry["grounded"] else "⚠️"
    combo = f"{entry['chunk_strategy']} | {entry['embedding_model']} | {entry['backend']} | {entry['scope']}"
    return (
        f"`{entry['timestamp']}` {grounded} **{combo}** — \n\"{entry['query']}\"\n"
        f"{entry['answer']}\n"
        f"{citations}"
    )


def run_query(
    query, chunk_strategy, embedding_model, backend, scope, chunks_top_k, max_tokens, history,
    request: gr.Request = None, session_id: str = None,
):
    embedder = get_embedder(embedding_model)
    name = collection_name(chunk_strategy, embedding_model, scope)
    # session_hash is a fresh random id per browser tab load (regenerated on refresh), so it
    # naturally maps to "one sitting" — grouping this tab's queries into one Langfuse session.
    # `session_id` lets callers outside a Gradio request (e.g. seed_demo_metrics.py) supply
    # their own, so separate script runs don't all collapse into one indistinguishable session.
    if session_id is None:
        session_id = request.session_hash if request is not None else "cli"
    try:
        with propagate_attributes(session_id=session_id):
            result = ask(query, name, embedder, embedding_model, backend, chunks_top_k=chunks_top_k, max_tokens=max_tokens)
    except Exception:
        logfire.exception("ask failed", backend=backend, embedding_model=embedding_model, collection_name=name)
        errors_total.add(1, {"backend": backend})
        result = {
            "answer": "Something went wrong answering this query. Check the logs for details.",
            "citations": [],
            "grounded": False,
            "trace_id": None,
        }
    entry = {
        "query": query,
        "chunk_strategy": chunk_strategy,
        "embedding_model": embedding_model,
        "backend": backend,
        "scope": scope,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        **result,
    }
    history = history + [entry]
    log_md = "\n\n---\n\n".join(format_entry(e) for e in reversed(history))
    return history, log_md


def start_loading():
    return gr.update(value="Asking...", interactive=False)


def stop_loading():
    return gr.update(value="Ask", interactive=True)


def show_session(request: gr.Request) -> str:
    """Displayed so the demo audience can see which Langfuse session this tab's queries land in."""
    return f"Session: `{request.session_hash[:8]}` (a new one is assigned every time this page reloads)"


def record_feedback(positive: bool, history: list) -> str:
    """Score the most recent answer's trace with a thumbs up/down, visible in Langfuse under that trace's scores."""
    if not history:
        return "Ask a question first."
    trace_id = history[-1].get("trace_id")
    if not trace_id:
        return "No trace to attach feedback to for that answer."
    langfuse.create_score(trace_id=trace_id, name="user_feedback", value=1 if positive else 0, data_type="BOOLEAN")
    return "Thanks for the feedback!" if positive else "Thanks — noted as not helpful."


with gr.Blocks(title="RAG mission") as demo:
    gr.Markdown("# RAG mission - Querying Hugging Face")
    session_display = gr.Markdown()
    demo.load(fn=show_session, outputs=session_display)
    scopes = list_built_scopes()
    history_state = gr.State([])

    with gr.Row():
        query = gr.Textbox(label="ask", placeholder="Ask a question about the Hugging Face docs...", scale=4)
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        chunk_strategy = gr.Radio(
            CHUNK_STRATEGIES,
            value="recursive",
            label="Chunk strategy",
            info="Recursive splits text into smaller chunks recursively and semantic chunking splits text using an llm.",
        )
        embedding_model = gr.Radio(
            EMBEDDING_MODELS,
            value="minilm",
            label="Embedding model",
            info="minilm is a smaller, faster model; bge is a larger, more accurate model.",
        )
        embedding_model.change(fn=warm_embedder, inputs=embedding_model, outputs=[])
        backend = gr.Radio(
            BACKENDS,
            value=BACKENDS[0],
            label="Backend",
            info="Which LLM generates the answer from the retrieved chunks (local Gemma vs. Claude API).",
        )
        backend.change(fn=warm_backend, inputs=backend, outputs=[])
        scope = gr.Radio(
            scopes, value=scopes[0], label="Scope", info="How much of the corpus was indexed."
        )

    with gr.Row():
        chunks_top_k = gr.Slider(
            minimum=1,
            maximum=20,
            value=DEFAULT_CHUNKS_TOP_K,
            step=1,
            label="Chunks retrieved (k)",
            info="How many chunks retrieval hands to the LLM — more gives the answer a wider net, but also more noise.",
        )
        max_tokens = gr.Slider(
            minimum=50,
            maximum=1000,
            value=MAX_NEW_TOKENS,
            step=50,
            label="Max response tokens",
            info="Upper bound on how long the generated answer can be.",
        )

    log = gr.Markdown(label="History")

    with gr.Row():
        thumbs_up = gr.Button("👍 Good answer", size="sm")
        thumbs_down = gr.Button("👎 Not helpful", size="sm")
        feedback_status = gr.Markdown()

    ask_btn.click(fn=start_loading, outputs=ask_btn).then(
        fn=run_query,
        inputs=[query, chunk_strategy, embedding_model, backend, scope, chunks_top_k, max_tokens, history_state],
        outputs=[history_state, log],
    ).then(fn=stop_loading, outputs=ask_btn)

    thumbs_up.click(fn=lambda history: record_feedback(True, history), inputs=history_state, outputs=feedback_status)
    thumbs_down.click(fn=lambda history: record_feedback(False, history), inputs=history_state, outputs=feedback_status)

if __name__ == "__main__":
    warm_embedder("minilm")
    warm_backend(BACKENDS[0])
    demo.launch()
