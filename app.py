"""app.py — Gradio UI for comparing chunk_strategy x embedding_model x backend combinations via a history log."""

from dotenv import load_dotenv

load_dotenv()

import logfire

# rag-hf-docs sends telemetry to its own Logfire project (us region); LOGFIRE_TOKEN in .env picks the project.
logfire.configure(advanced=logfire.AdvancedOptions(base_url="https://logfire-us.pydantic.dev"))
logfire.instrument_anthropic()  # covers the "api" backend; the local HF backend gets a manual span (see generation.py)

from datetime import datetime

import gradio as gr

from ask import ask
from embeddings import get_embedder
from generation import LOCAL_MODEL_ID, warm_local_pipeline
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
        f"{entry['answer']}\n\n"
        f"{citations}"
    )


def run_query(query, chunk_strategy, embedding_model, backend, scope, history):
    embedder = get_embedder(embedding_model)
    name = collection_name(chunk_strategy, embedding_model, scope)
    result = ask(query, name, embedder, embedding_model, backend)
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


with gr.Blocks(title="RAG mission") as demo:
    gr.Markdown("# RAG mission - Querying Hugging Face")
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

    log = gr.Markdown(label="History")

    ask_btn.click(fn=start_loading, outputs=ask_btn).then(
        fn=run_query,
        inputs=[query, chunk_strategy, embedding_model, backend, scope, history_state],
        outputs=[history_state, log],
    ).then(fn=stop_loading, outputs=ask_btn)

if __name__ == "__main__":
    warm_embedder("minilm")
    warm_backend(BACKENDS[0])
    demo.launch()
