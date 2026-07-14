"""app.py — Gradio UI for comparing chunk_strategy x embedding_model x backend combinations via a history log."""

from dotenv import load_dotenv

load_dotenv()

import logfire

# rag-hf-docs sends telemetry to its own Logfire project (us region); LOGFIRE_TOKEN in .env picks the project.
logfire.configure(advanced=logfire.AdvancedOptions(base_url="https://logfire-us.pydantic.dev"))
logfire.instrument_anthropic()  # covers the "api" backend; the local HF backend gets a manual span (see generation.py)

import gradio as gr

from ask import ask
from embeddings import get_embedder
from vectorstore import collection_name, list_built_scopes

CHUNK_STRATEGIES = ["recursive", "semantic"]
EMBEDDING_MODELS = ["minilm", "bge"]
BACKENDS = ["local", "api"]


def format_entry(entry: dict) -> str:
    citations_md = "\n".join(f"- [{c['marker']}] {c['source']}" for c in entry["citations"]) or "_none_"
    grounded_md = "✅ Grounded" if entry["grounded"] else "⚠️ Not grounded"
    combo = f"{entry['chunk_strategy']} / {entry['embedding_model']} / {entry['backend']} / {entry['scope']}"
    return (
        f"**{combo}** — \"{entry['query']}\"\n\n"
        f"{entry['answer']}\n\n"
        f"**Citations:**\n{citations_md}\n\n"
        f"{grounded_md}"
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
        **result,
    }
    history = history + [entry]
    log_md = "\n\n---\n\n".join(format_entry(e) for e in reversed(history))
    return history, log_md


with gr.Blocks(title="RAG comparison") as demo:
    gr.Markdown("# Hugging Face Docs RAG — history log for comparing combinations")
    scopes = list_built_scopes()
    history_state = gr.State([])

    with gr.Row():
        query = gr.Textbox(label="Question", scale=4)
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        chunk_strategy = gr.Dropdown(
            CHUNK_STRATEGIES,
            value="recursive",
            label="Chunk strategy",
            info="How documents were split into chunks, before you ever open this app — fixed at build time, not affected by anything below.",
        )
        embedding_model = gr.Dropdown(
            EMBEDDING_MODELS,
            value="minilm",
            label="Embedding model",
            info="This is your retrieval choice — the only search method here is vector similarity, and this picks which vector space it searches in.",
        )
        backend = gr.Dropdown(
            BACKENDS,
            value="local",
            label="Backend",
            info="Which LLM generates the answer from the retrieved chunks (local Gemma vs. Claude API).",
        )
        scope = gr.Dropdown(
            scopes, value=scopes[0], label="Scope", info="How much of the corpus was indexed."
        )

    log = gr.Markdown(label="History")

    ask_btn.click(
        fn=run_query,
        inputs=[query, chunk_strategy, embedding_model, backend, scope, history_state],
        outputs=[history_state, log],
    )

if __name__ == "__main__":
    demo.launch()
