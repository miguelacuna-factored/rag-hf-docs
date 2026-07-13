"""app.py — Gradio UI for comparing chunk_strategy x embedding_model x backend combinations side by side."""

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


def run_query(query, chunk_strategy, embedding_model, backend, scope):
    embedder = get_embedder(embedding_model)
    name = collection_name(chunk_strategy, embedding_model, scope)
    result = ask(query, name, embedder, embedding_model, backend)
    citations_md = "\n".join(f"- [{c['marker']}] {c['source']}" for c in result["citations"]) or "_none_"
    grounded_md = "✅ Grounded" if result["grounded"] else "⚠️ Not grounded"
    return result["answer"], citations_md, grounded_md


def build_column():
    with gr.Column():
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
        answer = gr.Markdown(label="Answer")
        citations = gr.Markdown(label="Citations")
        grounded = gr.Markdown(label="Grounded")
    return chunk_strategy, embedding_model, backend, answer, citations, grounded


with gr.Blocks(title="RAG comparison") as demo:
    gr.Markdown("# Hugging Face Docs RAG — side-by-side comparison")
    scopes = list_built_scopes()
    with gr.Row():
        query = gr.Textbox(label="Question", scale=4)
        scope = gr.Dropdown(
            scopes, value=scopes[0], label="Scope", info="How much of the corpus was indexed.", scale=1
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        left = build_column()
        right = build_column()

    for chunk_strategy, embedding_model, backend, answer, citations, grounded in (left, right):
        ask_btn.click(
            fn=run_query,
            inputs=[query, chunk_strategy, embedding_model, backend, scope],
            outputs=[answer, citations, grounded],
        )

if __name__ == "__main__":
    demo.launch()
