# GCP Analytics RAG

Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course, built to run on a laptop
within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage, `multilingual-e5-small`
(int8, ONNX Runtime) for embeddings.

Status: ingestion and hybrid retrieval work (vector + keyword legs fused with RRF). Evaluation and the MCP server are next.

## Layout

```
src/rag/
├── embed.py          # ONNX embedder (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── chunk.py      # header-aware chunking
│   └── pipeline.py   # incremental ingest (hash-based)
└── retrieval/        # online path
    ├── fusion.py     # Reciprocal Rank Fusion
    └── search.py     # HybridRetriever: vector leg + keyword leg
tests/                # mirrors src/
infra/                # container definitions
data/  models/        # local only, git-ignored (course files, ONNX model)
```

## Run

Database (Podman; `podman compose` does not work here, use `podman run`):

```bash
podman run -d --name rag-db -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -p 127.0.0.1:5433:5432 -v rag_pgdata:/var/lib/postgresql/data --memory 1g --cpus 2 \
  docker.io/pgvector/pgvector:pg17
```

Python (uv, Python 3.12):

```bash
uv sync
uv run pytest
```

The embedding model goes in `models/multilingual-e5-small/` (`onnx/model_quantized.onnx` and
`tokenizer.json` from `Xenova/multilingual-e5-small` on Hugging Face).
