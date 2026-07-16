# forest-rag

A production-oriented RAG (Retrieval-Augmented Generation) pipeline that answers questions on the agriculture volume of [The Shift Project](https://theshiftproject.org/) reports.

## Purpose

A RAG system that answers questions grounded in a fixed, domain-specific corpus, built and deployed end-to-end rather than left as a notebook prototype. Scope and stack are kept deliberately small — no component included unless it solves an actual problem in this pipeline.

## Architecture

```
PDF reports → chunking → dense embeddings → Qdrant (vector store)
                                                   │
User query → embed query → retrieve top-k → rerank (cross-encoder) → LLM (Groq) → answer
                                                   │
                                            Langfuse (tracing/observability)
```

## Tech stack

| Component | Choice | Notes |
|---|---|---|
| API framework | FastAPI | async, lifespan-managed model loading |
| Vector store | Qdrant (embedded mode, `path=`) | local, no server process required |
| Embeddings | fastembed `TextEmbedding` | dense embeddings |
| Reranking | `jinaai/jina-reranker-v2-base-multilingual` (cross-encoder, via fastembed) | `threads=1` to control memory footprint |
| LLM | Groq API, `llama-3.3-70b-versatile` | free tier compatible with a portfolio/demo deployment |
| Observability | Langfuse | trace generation calls, inspect retrieved context |
| Dependency management | Poetry | |
| Deployment target | Hugging face space | |

No orchestration framework (LangChain, etc.). Retrieval, reranking, and generation are called directly — the pipeline is a fixed sequence, not a graph that needs an abstraction layer.

## Retrieval pipeline

1. Source PDFs are chunked and embedded with fastembed.
2. Chunks are stored in Qdrant (embedded/local mode — `path="db"`, not a hosted `location=`).
3. At query time, the query is embedded, top-k candidates are retrieved from Qdrant, then reranked with the Jina cross-encoder to improve precision over the initial bi-encoder retrieval.
4. Reranked context is injected into a structured prompt sent to Groq (`llama-3.3-70b-versatile`, `temperature=0.1` for grounded output).

### Prompt structure

System role → task instructions → retrieved chunks (XML-delimited) → grounding reminder → user question (last).

## Project structure

```
app/            # FastAPI app, routers, lifespan (model/index loading)
scripts/
  ingestion/    # PDF parsing (PyMuPDF) and chunking
db/             # local Qdrant embedded storage
tests/          # pytest + pytest-asyncio + asgi-lifespan smoke tests
```

## Running locally

```bash
poetry install
poetry run uvicorn app.main:app --reload
```

Run from the project root — modules are resolved as `app.*` / `scripts.*` (e.g. `python -m scripts.ingestion.splitting`), not via relative script execution.

## Testing

Smoke tests use `asgi_lifespan.LifespanManager` to exercise the FastAPI lifespan (model + index loading) under `pytest-asyncio`.

```bash
poetry run pytest
```

## Known constraints

- Embedding model + reranker (ONNX) + Qdrant loaded at startup are memory-intensive; on constrained environments (e.g. 8GB RAM under WSL2), reranker thread count is capped (`threads=1`) to avoid OOM.
- Free hosting rules out self-hosting an LLM, so inference runs on Groq's API rather than a locally-served model. This caps throughput and model choice but keeps the deployment simple and cheap to run.

## Status

- [x] Qdrant migration (from Chroma)
- [x] Ingestion pipeline
- [x] Retrieval + reranking
- [x] OOM resolution under WSL2 
- [x] Smoke test validation
- [ ] Deployment to Render

## License

TBD.