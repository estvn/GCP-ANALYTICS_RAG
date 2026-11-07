# GCP Analytics RAG

Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course, built to run on a laptop
within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage, `multilingual-e5-small`
(int8, ONNX Runtime) for embeddings.

Status: ingestion, hybrid retrieval and evaluation work. The MCP server is next.

**This branch (`chunking/semantic`) contains only the semantic chunking technique.** See [CHUNKING.md](CHUNKING.md).

## Layout

```
src/rag/
├── embed.py          # ONNX embedder (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # semantic chunking under a token cap
│   └── pipeline.py   # incremental ingest (hash-based)
├── retrieval/        # online path
│   ├── fusion.py     # Reciprocal Rank Fusion
│   └── search.py     # HybridRetriever: vector leg + keyword leg, multi-phrasing search
└── evaluation/       # golden sets, hit@k / MRR, comparison report
eval/                 # golden sets and results.md
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

## Results

On exact-term queries hybrid retrieval reaches hit@5 1.00; on paraphrased questions hybrid and vector-only are
equivalent (about 0.70 hit@5 on two independent sets of 40). Within a language it answers 26 of 26 questions of the
first set; the weak spot is cross-language search, handled by querying in both languages
(`HybridRetriever.search_many`, hit@5 0.925). Structural, semantic and structure-first chunking tie within the noise
of these sets, so the simplest one is the default. Full tables, what each experiment changed and the limits of the
evaluation are in [eval/results.md](eval/results.md). Reproduce with:

```bash
uv run python -m rag.evaluation.report
```
