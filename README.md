# Semantic chunking

Branch `chunking/semantic-e5-small`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Embedding model of
this branch's results: `e5-small` (int8, ONNX Runtime).

Full description, settings and how to reproduce: [CHUNKING.md](CHUNKING.md). Raw evaluation lines:
[eval/runs/semantic-e5-small.log](eval/runs/semantic-e5-small.log).

## Technique applied

1. Split the text into units (sentences, bullets, wrapped lines); a header travels with the first unit after it.
2. Embed each unit together with its neighbours (previous, current, next) with the retrieval model.
3. Cut where the distance between consecutive windows is above the 90th percentile of the document's distances.
4. A chunk is never below 100 tokens (small ones are merged) and never above 440 tokens. Headers are context only,
   not cut points.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E14**: Boundary quality audits (what gets cut)
- **E15**: Semantic chunker, round 1
- **E16**: Semantic vs structural at equal returned text
- **E17**: Grid with a validation set: 6 chunkers x 7 search variants
- **E19**: Full-index evaluation of the three chunkers
- **E29**: All chunking techniques on one common method
- **E31**: Returning 30 chunks and a larger candidate pool

## Results (E29: hybrid search, 130 questions, hits / MRR)

| technique | e5-small | bge-m3 |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 **◀ this branch** | 113 / .576 |
| semantic + header cuts | 99 / .557 | 114 / .580 |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

Hits are questions answered out of 130, with about 9,000 characters returned per query. Differences under about 8 points
(about 10 of the 130 questions) are not conclusive.

## Results returning 30 chunks (E31)

Index `full_SEM` (`e5-small`), the same 130 questions; each query returns up to 30 chunks (about 31,363 characters). Questions answered out of 130 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search (the pool of chunks each search contributes before fusion):

| retrieval technique | @10 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| vector | 93 | 104 | 93 | 100 | 104 | 0.472 |
| vector + ES/EN 1:1 | 99 | 112 | 76 | 104 | 113 | 0.325 |
| vector + ES/EN 2:1 | 90 | 104 | 92 | 103 | 107 | 0.341 |
| keyword | 79 | 96 | 79 | 93 | 96 | 0.379 |
| keyword + ES/EN 1:1 | 78 | 104 | 75 | 90 | 103 | 0.302 |
| keyword + ES/EN 2:1 | 78 | 98 | 77 | 96 | 99 | 0.323 |
| hybrid | 97 | 109 | 97 | 104 | 108 | 0.523 |
| hybrid + ES/EN 1:1 | 106 | 121 | 89 | 112 | 120 | 0.367 |
| hybrid + ES/EN 2:1 | 97 | 109 | 98 | 110 | 112 | 0.420 |

- Best technique on this index: hybrid + ES/EN 1:1, 120 of 130 answered with 100 candidates (MRR 0.367); still missing: ['q35', 'v02', 'v05', 'v24', 'v29', 'v31', 'v35', 'f12', 'f38', 'f40'].
- At 20 chunks (about 20,909 characters) the best technique answers 112 of 130 with 100 candidates.
- Returning 30 chunks instead of about 10 answers more questions in every technique. A pool of 100 candidates helps at 30 chunks but hurts the top 10: plain hybrid goes from 97 to 97 answered at 10 chunks.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive. The 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log) and [eval/runs/e31_k30_candidates30.log](eval/runs/e31_k30_candidates30.log).

## Layout

```
src/rag/
├── embed.py          # ONNX embedders: bge-m3 and e5-small (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # SemanticChunker: semantic cuts only (headers are context, not cut points)
│   └── pipeline.py   # incremental ingest (hash-based)
├── retrieval/        # online path
│   ├── fusion.py     # Reciprocal Rank Fusion
│   └── search.py     # HybridRetriever: vector leg + keyword leg, search_many (question + translation)
└── evaluation/       # golden sets, hit rate / MRR, comparison report
eval/                 # golden sets, results.md, experiments.md, runs/ (raw output)
tests/                # mirrors src/
infra/                # container definitions
data/  models/        # local only, git-ignored (course files, ONNX models)
```

## Branches

Each technique that was compared has a branch with a README that explains it and what happened in its experiments.

| technique | branches |
|---|---|
| structural chunking | `chunking/structural-e5-small`, `chunking/structural-bge-m3` |
| semantic chunking | `chunking/semantic-e5-small`, `chunking/semantic-bge-m3` |
| semantic + header cuts (called `structure-semantic`) | `chunking/structure-semantic-e5-small`, `chunking/structure-semantic-bge-m3` |
| community two-stage chunking | `chunking/structure-then-semantic-e5-small`, `chunking/structure-then-semantic-bge-m3` |
| two-stage + merge (the author's variant) | `chunking/structure-semantic-merge-e5-small`, `chunking/structure-semantic-merge-bge-m3` |
| vector, keyword, hybrid search | `retrieval/vector`, `retrieval/keyword`, `retrieval/hybrid` |
| ES/EN fusion | `retrieval/vector-es-en-rrf`, `retrieval/hybrid-es-en-rrf` |

## Study guides (Spanish)

- [docs/01_chunking_embeddings_y_vectores.md](docs/01_chunking_embeddings_y_vectores.md): indexing, chunking techniques, embeddings, vectors and dimensions.
- [docs/02_tecnicas_de_retrieval.md](docs/02_tecnicas_de_retrieval.md): vector search, keyword search, RRF, multi-language fusion, model mismatch.
- [docs/03_arquitecturas_rag.md](docs/03_arquitecturas_rag.md): RAG architectures, what "RAG 1.0 / 2.0" mean.
- [docs/04_este_proyecto.md](docs/04_este_proyecto.md): this project: techniques, tools, advantages and limits.
