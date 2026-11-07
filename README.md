# Two-stage chunking (community recipe: headers, then semantic)

Branch `chunking/structure-then-semantic-e5-small`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Embedding model of
this branch's results: `e5-small` (int8, ONNX Runtime).

Full description, settings and how to reproduce: [CHUNKING.md](CHUNKING.md). Raw evaluation lines:
[eval/runs/structure-then-semantic-e5-small.log](eval/runs/structure-then-semantic-e5-small.log).

## Technique applied

1. **Structural stage:** split by markdown headers; consecutive sections below 100 tokens are merged, never past 440.
2. **Semantic stage:** only a section above 440 tokens is subdivided by the semantic cut chunker (90th percentile).
3. Small sections are never embedded to be cut, so it is cheaper than the one-pass chunkers. This is the community
   recipe; the user's hybrid is this plus a merge step (see `structure-semantic-merge-*`).

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E29**: All chunking techniques on one common method

## Results (E29: hybrid search, 130 questions, hits / MRR)

| technique | e5-small | bge-m3 |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | 114 / .580 |
| community two-stage | 97 / .508 **◀ this branch** | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

Hits are questions answered out of 130, with about 9,000 characters returned per query. A gap of 2 to 4 questions is
below the standard error (about 4 points): it is not conclusive.

## Layout

```
src/rag/
├── embed.py          # ONNX embedder (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── chunk.py      # header-aware chunking (default)
│   ├── semantic.py   # semantic chunking under a token cap (option)
│   └── pipeline.py   # incremental ingest (hash-based)
├── retrieval/        # online path
│   ├── fusion.py     # Reciprocal Rank Fusion
│   └── search.py     # HybridRetriever: vector leg + keyword leg, multi-phrasing search
└── evaluation/       # golden sets, hit@k / MRR, comparison report
eval/                 # golden sets, results.md and experiments.md (full ledger)
docs/                 # study guides (Spanish)
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
of these sets, so the simplest one is the default. The complete experiment ledger is [eval/experiments.md](eval/experiments.md). Full tables, what each experiment changed and the limits of the
evaluation are in [eval/results.md](eval/results.md). Reproduce with:

```bash
uv run python -m rag.evaluation.report
```

## Study guides (Spanish)

- [docs/01_chunking_embeddings_y_vectores.md](docs/01_chunking_embeddings_y_vectores.md): indexing, chunking techniques, embeddings, vectors and dimensions.
- [docs/02_tecnicas_de_retrieval.md](docs/02_tecnicas_de_retrieval.md): vector search, keyword search, RRF, multi-language fusion, model mismatch.
- [docs/03_arquitecturas_rag.md](docs/03_arquitecturas_rag.md): RAG architectures, what "RAG 1.0 / 2.0" mean.
- [docs/04_este_proyecto.md](docs/04_este_proyecto.md): this project: techniques, tools, advantages and limits.
