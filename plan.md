# RAG Pipeline Plan — Hugging Face Docs

Domain/corpus: Hugging Face documentation (`m-ric/huggingface_doc`, `train` split).

Goal: build a modular, end-to-end RAG pipeline that lets us compare **2 chunking strategies × 2 embedding models** (4 combinations), retrieve with citations, generate grounded answers, and report retrieval-quality numbers.

This is a teaching demo, not production code — a junior dev with no prior RAG exposure should be able to read one file at a time and understand what that stage of the pipeline is doing and why. Prioritize clarity over cleverness in every stage below.

## 0.a Reading order (for a first-time reader)

The file layout in Section 1 is listed in *build* order. To actually learn the pipeline, read/run in this order — each step's output is the next step's input:

1. `ingest.py` — turn raw dataset rows into `Document`s
2. `chunking.py` — split `Document`s into retrievable pieces
3. `embeddings.py` — turn text into vectors
4. `vectorstore.py` — store/index those vectors
5. `retrieval.py` — pull relevant chunks back out for a query
6. `generation.py` — turn retrieved chunks + query into a cited answer
7. `pipeline.py` — wire 5+6 together end-to-end
8. `main.py` — the CLI entrypoint a reader actually runs first
9. `evaluate.py` — measure how good steps 2-5 are, across combinations

This is also the build order used in Section 1's file layout and the section numbering below — `evaluate.py` is listed and numbered last in both places since it only depends on `retrieval.py`, not on `pipeline.py`/`main.py`, but is easiest to read once the whole pipeline is in view.

Put this ordering in the README so a reader isn't left to infer it from file listing order.

## 0.b Code conventions for this repo

These apply to every module below; call them out explicitly to whoever implements this plan (including an LLM) so style doesn't drift file-to-file:

- **Plain functions over classes.** Every stage is a function (or a small set of them), not a class hierarchy. Do not introduce an abstract base class, protocol, or "strategy pattern" object for chunkers/embedders/backends just because there are two of each — a `chunk_strategy: str` argument plus an if/dict-dispatch is easier for a junior dev to trace than a class hierarchy, and there's no third implementation coming that would justify the abstraction. This is SOLID's Single Responsibility Principle applied at the *function/module* level (one file = one pipeline stage), not at the class level.
- **Linear orchestration, dependencies passed explicitly.** `main.py` (and `pipeline.py` for the query path) is the only place that wires stages together: it imports every stage's function and calls them in order — `load_raw_dataset` → `load_documents` → chunker → `build_collection` → `search` → `generate` — matching the reading order in §0.a. A stage module never imports and calls the *previous* stage's module itself (e.g. `ingest.py` must not import `dataset.py`); instead every stage function takes its inputs as explicit parameters (raw rows, a `Document` list, an embedder instance, etc.) and the orchestrator threads outputs into the next call as arguments. This is dependency injection in its cheapest form — plain parameter passing, no container, no constructor-injected interfaces/protocols — and it's what actually gives us the "swap a stage without touching its neighbors" property: a stage can be unit-tested or reordered because it never reaches out to fetch its own inputs. Heavier DI (service registries, runtime-resolved interfaces) is not needed for a one-shot CLI pipeline and is explicitly deferred.
- **Type hints and one-line docstrings on every public function.** Signature + one sentence of *why*, not *what* (the code already says what). E.g. `def search(query: str, collection_name: str, k: int = 5) -> list[Document]:` with a docstring noting the BGE prefix caveat, not restating "does a search."
- **One vocabulary, used identically everywhere.** Don't let terms drift between sections/files. Use exactly:
  - `scope`: `"subset"` | `"full"`
  - `chunk_strategy`: `"recursive"` | `"semantic"`
  - `embedding_model`: `"minilm"` | `"bge"` (short keys used in code/collection names; full HF model ids live only in `embeddings.py`)
  - `backend`: `"local"` | `"api"`
  - `collection_name`: always built as `f"{chunk_strategy}_{embedding_model}_{scope}"` — build it with one shared helper (e.g. in `vectorstore.py`) rather than re-concatenating the string in multiple files.
- **Consistent data shape between stages.** Every stage after ingestion takes and returns `Document` (or `list[Document]`) until `generation.py`, which returns the `{answer, citations, grounded}` dict from Section 7. Don't introduce ad hoc tuples/dicts as intermediate shapes.
- **Module-level docstring explaining the concept, not just the code.** Since this is a teaching artifact, each file should open with 2-3 lines answering "what RAG concept does this file demonstrate, and why does it matter" (e.g. `chunking.py`: why chunk size/overlap affects retrieval quality) before diving into implementation.

## 0.c Caching policy

We'll run this on multiple laptops and re-run it many times while iterating — nothing expensive (model download, dataset download, chunking, embedding) should ever redo work that's already on disk from a previous run on that machine. Three distinct caches, each already implied by earlier sections — this just makes the rule explicit and gives it one shared implementation instead of three ad hoc ones:

- **HF model/dataset downloads** — handled automatically by `transformers`/`sentence-transformers`/`datasets`' own on-disk cache (`~/.cache/huggingface`). No code needed for this one; just don't pass `force_download=True` or a throwaway `cache_dir` anywhere.
- **Chunked documents & the eval set** (Section 3, Section 9) — small file-based cache under `./data/`. Add one small shared helper, `load_cached_or_build(path: Path, build_fn: Callable[[], T], loader, saver) -> T` (a natural home is a new `cache.py`, or a few lines at the top of `chunking.py` if it feels too small for its own file — implementer's call), used by both chunking and eval-set generation instead of each hand-rolling its own "if file exists, load; else, compute and save" block. Keep it a plain function, not a decorator or class — same "plain functions" convention as the rest of the repo.
- **Vector store collections** (Section 5) — already covered by `build_collection`'s `collection.count()` check; no separate file cache needed here since Chroma's own persistence at `./vectordb` *is* the cache.

Make sure `./data/` and `./vectordb` are listed in `.gitignore` (they're regenerable local caches, not source).

## 0. Dependencies to add

Current `pyproject.toml` has `langchain`, `langchain-community`, `sentence-transformers`, `datasets`, `torch`, `transformers`, `accelerate` — but is missing pieces we need:

- `chromadb` — vector store
- `langchain-chroma` — LangChain's Chroma integration
- `langchain-huggingface` — modern `HuggingFaceEmbeddings` (the old one lives in `langchain-community` and is deprecated)
- `langchain-experimental` — for `SemanticChunker`
- `langchain-text-splitters` — for `RecursiveCharacterTextSplitter` (may already be a transitive dep of `langchain`, confirm)
- `openai` — used only as the OpenRouter-compatible client for the API generation backend (OpenRouter exposes an OpenAI-compatible endpoint)

Retrieval evaluation (Stage 9) uses manual metrics (Hit Rate@k, MRR@k, Precision@k) computed with plain Python/pandas — no `ragas` or other eval library needed.

`accelerate` is already present and required as-is: it's what makes `device_map="auto"` in `generation.py` (Stage 7) resolve to MPS on the M1 / CUDA on the RTX 5080.

Not used anywhere in this plan and candidates to remove once implementation starts (confirm with the user first): `bitsandbytes` (CUDA-only quantization, not needed since Gemma 2 2B-it runs unquantized in fp16/bf16 on both target machines), `faiss-cpu` (superseded by Chroma), `ragatouille`, `pacmap`, `openpyxl` (leftover from an earlier notebook, unrelated to this pipeline).

Action: `uv add chromadb langchain-chroma langchain-huggingface langchain-experimental openai` once we start implementing (not yet).

## 1. Module layout

Keep each rubric-required stage in its own file so ingestion/retrieval/generation are independently swappable and testable:

```text
rag-hf-docs/
├── dataset.py          # (existing) raw dataset loading only — one function, see Section 2
├── ingest.py          # calls dataset.py, then converts rows to LangChain Documents
├── chunking.py         # two chunkers: recursive + semantic, both return list[Document]
├── embeddings.py       # factory for the two embedding models
├── vectorstore.py      # builds/loads the Chroma collections (4 or 8, depending on scope)
├── retrieval.py        # semantic search over a given collection
├── generation.py       # prompt template + LLM call + citation formatting
├── pipeline.py         # orchestrates end-to-end query -> retrieve -> generate -> cited answer
├── main.py             # CLI entrypoint (build index | ask a question | run eval)
└── evaluate.py         # builds a small eval set, scores retrieval quality per combo
```

This keeps "modular ingestion, retrieval, and generation stages" explicit per the rubric wording.

## 2. Stage: Ingestion (`ingest.py`)

- `dataset.py` stays the single place that loads the raw HF dataset — refactor its current top-level script (which calls `load_dotenv()` and `load_dataset("m-ric/huggingface_doc", split="train")` at module scope) into one function, `load_raw_dataset() -> Dataset`, wrapping that same call, so nothing runs on import. Per the linear-orchestration convention (§0.b), `ingest.py` does **not** import `dataset.py` or call `load_raw_dataset` itself — `main.py` calls `load_raw_dataset()` and passes the resulting rows into `load_documents(rows, scope)` as a parameter. This is the DRY boundary: "load raw rows" (`dataset.py`) vs. "shape rows into `Document`s for this pipeline" (`ingest.py`) stay two separate, single-purpose functions instead of one merged or two duplicated, and neither reaches into the other's module to get there.
- `scope` parameter (`"subset"` | `"full"`): `subset` takes the first N documents (e.g. N=200, tune after a timing dry-run) for fast iteration; `full` uses all ~2,200 docs. Same ingestion code path either way — just a slice before returning.
- Convert each HF dataset row into a LangChain `Document(page_content=row["text"], metadata={"source": row["source"], ...})`. Check what metadata fields the dataset actually exposes (likely `source` = doc URL/path) — inspect `dataset.column_names` first.
- Metadata is critical: it's what citations are built from later, so capture a stable identifier per document (URL or path) at ingestion time, before chunking, and propagate it into every chunk's metadata plus a chunk index.

## 3. Stage: Chunking (`chunking.py`)

Two strategies, both operating on the same ingested `Document` list, each tagged with a strategy name in metadata so we can filter/compare later:

1. **Recursive Character Splitter (baseline)** — `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`. Fast, deterministic, no model dependency.
2. **Semantic Chunking (advanced)** — `langchain_experimental.text_splitter.SemanticChunker`, using one of the embedding models (likely MiniLM, since it's cheaper) to detect breakpoints by embedding similarity between sentences.

Each chunk gets metadata: `{source, chunk_strategy: "recursive"|"semantic", scope: "subset"|"full", chunk_id}`. Output: lists of `Document` chunks per (chunker × scope), cached under `./data/` via the `load_cached_or_build` helper from Section 0.c so re-running `main.py build` doesn't re-chunk (and, for `SemanticChunker`, re-embed) work already on disk.

## 4. Stage: Embeddings (`embeddings.py`)

Factory function returning a `HuggingFaceEmbeddings` instance for:

- `sentence-transformers/all-MiniLM-L6-v2` (lightweight, 384-dim)
- `BAAI/bge-base-en-v1.5` (advanced, 768-dim — note BGE models recommend a query instruction prefix like `"Represent this sentence for searching relevant passages: "` for queries, not documents; need to wire that in for retrieval quality)

## 5. Stage: Vector store (`vectorstore.py`)

- Local persistent Chroma client at `./vectordb`.
- Collections are named by all three axes so `subset` and `full` builds coexist without clobbering each other: `{chunk_strategy}_{embedding_model}_{scope}`, e.g. `recursive_minilm_subset`, `semantic_bge_full`. Running only `subset` gives 4 collections; running both `subset` and `full` gives 8.
- Build function: `build_collection(chunks, embedding_model, collection_name)` — idempotent, skips rebuild if collection already populated (checked via `collection.count()`), so re-running `main.py` doesn't re-embed everything.

## 6. Stage: Retrieval (`retrieval.py`)

- `search(query, collection_name, k=5) -> list[Document]` wrapping Chroma's similarity search.
- Apply the BGE query-instruction prefix here when the collection uses BGE embeddings.
- Return documents with metadata intact (source, chunk_strategy) for citation building downstream.
- An empty result is a legitimate outcome, not an error — `search` just returns `[]`. `pipeline.py` (Section 8) is responsible for deciding what an empty result means for the answer; `search` itself doesn't guess. Note: plain Chroma similarity search returns the `k` nearest chunks regardless of how weak the match is, so `[]` in practice only happens when the collection has fewer than `k` documents — for a genuinely off-topic query, detecting "no good answer" relies on the LLM's "not in corpus" instruction (Section 7), not on this short-circuit. A `score_threshold` param on `search` would make the empty-result path fire for real, but is deferred until after eval numbers show whether it's actually needed.

## 7. Stage: Generation (`generation.py`)

- Prompt template that:
  - Includes retrieved chunks labeled with numbered citation markers, e.g. `[1] (source: ...)`.
  - Instructs the LLM to answer **only** from the provided context and to explicitly say "the answer isn't in the corpus" if the context doesn't support an answer.
  - Requires inline citation markers (`[1]`, `[2]`) in the generated answer, mapped back to source metadata for a final "Sources:" list.
- Two pluggable backends behind one interface (`generate(prompt) -> str`), selected via a `backend` param or env var:
  - **Local (default):** `google/gemma-2-2b-it` via `transformers`, loaded in fp16/bf16 with `device_map="auto"` (resolves to MPS on the M1, CUDA on the RTX 5080). No bitsandbytes/quantization needed — the model is small enough to fit both machines unquantized.
  - **API (optional):** Claude Haiku via OpenRouter, called through the `openai` SDK pointed at OpenRouter's OpenAI-compatible base URL (`https://openrouter.ai/api/v1`), with `OPENROUTER_API_KEY` read from `.env`. Model id e.g. `anthropic/claude-haiku-4.5` (confirm exact OpenRouter slug when implementing — check current model list, since aliases change).
- Output structure: `{answer: str, citations: [{marker, source}], grounded: bool}`. `grounded` is `True` only if the answer contains at least one citation marker that maps back to a retrieved chunk; it's `False` whenever the LLM outputs the "not in corpus" phrase (or the marker check fails) — this is a plain string/regex check in `generation.py`, not something the LLM self-reports.

## 8. Orchestration (`pipeline.py`, `main.py`)

- `pipeline.py`: `answer_question(query, collection_name, backend="local"|"api") -> cited answer`, chaining retrieval.py + generation.py. If `search` returns `[]`, skip `generation.py` entirely and return `{answer: "No relevant information found in the corpus.", citations: [], grounded: False}` directly — don't spend an LLM call asking it to notice there's no context.
- Per the linear-orchestration convention (§0.b), both illustrate the shape every stage call should take — flat, sequential, dependencies passed as arguments, no stage reaching into another stage's module:

  ```python
  # pipeline.py
  def answer_question(query, collection_name, backend="local"):
      chunks = search(query, collection_name)
      if not chunks:
          return {"answer": "No relevant information found in the corpus.", "citations": [], "grounded": False}
      return generate(query, chunks, backend)
  ```

  ```python
  # main.py (build subcommand, one combo shown)
  rows = load_raw_dataset()
  docs = load_documents(rows, scope)
  embedder = get_embedder(embedding_model)
  chunks = chunk_documents(docs, chunk_strategy, scope, embedder)
  build_collection(chunks, embedder, collection_name(chunk_strategy, embedding_model, scope))
  ```

- `main.py` CLI with subcommands: `build --scope subset|full` (run ingestion → chunking → embedding → vector store for the 4 combos in that scope), `ask --collection <name> --backend local|api "question"`, `eval --scope subset|full` (run evaluate.py, print/save comparison table).

## 9. Stage: Evaluation / comparison (`evaluate.py`)

Rubric requires "a comparison table showing chunking strategy vs retrieval quality with real numbers."

- Build a small labeled eval set (~20-30 question/answer pairs): sample chunks from the ingested corpus, prompt a top-tier API model (e.g. Claude, via OpenRouter or direct API) to generate a natural question that chunk answers plus the known ground-truth source, then do a quick manual pass to drop any low-quality/ambiguous pairs. Cache as `./data/eval_set.json` via the same `load_cached_or_build` helper (Section 0.c) so it's built once and reused across all collection combos and re-runs instead of regenerated (and re-billed, for the API call) every time.
- Metrics computed per (chunking strategy × embedding model × scope) combination, at fixed k:
  - **Hit Rate@k** — did any retrieved chunk come from the correct source doc?
  - **MRR@k** — rank of the first correct chunk.
  - **Precision@k** — fraction of retrieved chunks from a correct source.
- Produces a table (4 rows for `subset`-only, 8 rows if `full` is also built) with these three metrics, written to `./data/results.md` or printed via a small pandas DataFrame.
- Only exercises `retrieval.py` (steps 2-5 of the pipeline) — it does not need `generation.py` or `pipeline.py`, since it scores retrieval quality, not answer quality.

## 10. Write-up deliverable

One paragraph (per rubric) on what would change with a larger corpus — draft after eval numbers are in, informed by observed bottlenecks (e.g. semantic chunking cost scaling, embedding throughput, Chroma index size/latency, need for approximate search or a managed vector DB, need for a larger/curated eval set).

