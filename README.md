# Two-stage chunking (community recipe: headers, then semantic)

Branch `chunking/structure-then-semantic-bge-m3`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Embedding model of
this branch's results: `bge-m3` (int8, ONNX Runtime).

Full description, settings and how to reproduce: [CHUNKING.md](CHUNKING.md). Raw evaluation lines:
[eval/runs/structure-then-semantic-bge-m3.log](eval/runs/structure-then-semantic-bge-m3.log).

## Technique applied

1. **Structural stage:** split by markdown headers; consecutive sections below 100 tokens are merged, never past 440.
2. **Semantic stage:** only a section above 440 tokens is subdivided by the semantic cut chunker (90th percentile).
3. Small sections are never embedded to be cut, so it is cheaper than the one-pass chunkers. This is the community
   recipe; the user's hybrid is this plus a merge step (see `structure-semantic-merge-*`).

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E29**: All chunking techniques on one common method
- **E31**: Returning 30 chunks and a larger candidate pool

## Results (E29: hybrid search, 130 questions, hits / MRR)

| technique | e5-small | bge-m3 |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | 114 / .580 |
| community two-stage | 97 / .508 | 113 / .584 **◀ this branch** |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

Hits are questions answered out of 130, with about 9,000 characters returned per query. Differences under about 8 points
(about 10 of the 130 questions) are not conclusive.

## Results returning 30 chunks (E31)

Index `full_STSEM_bge` (`bge-m3`), the same 130 questions; each query returns up to 30 chunks (about 27,321 characters). Questions answered out of 130 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search (the pool of chunks each search contributes before fusion):

| retrieval technique | @10 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| vector | 107 | 120 | 107 | 116 | 120 | 0.602 |
| vector + ES/EN 1:1 | 105 | 121 | 106 | 117 | 122 | 0.603 |
| vector + ES/EN 2:1 | 106 | 120 | 107 | 119 | 122 | 0.597 |
| keyword | 80 | 98 | 80 | 94 | 98 | 0.359 |
| keyword + ES/EN 1:1 | 73 | 103 | 72 | 93 | 102 | 0.318 |
| keyword + ES/EN 2:1 | 76 | 100 | 80 | 96 | 102 | 0.332 |
| hybrid | 113 | 123 | 105 | 120 | 123 | 0.577 |
| hybrid + ES/EN 1:1 | 114 | 123 | 119 | 122 | 125 | 0.540 |
| hybrid + ES/EN 2:1 | 112 | 123 | 118 | 123 | 125 | 0.591 |

- Best technique on this index: hybrid + ES/EN 2:1, 125 of 130 answered with 100 candidates (MRR 0.591); still missing: ['q21', 'q40', 'v24', 'v28', 'f40'].
- At 20 chunks (about 18,214 characters) the best technique answers 123 of 130 with 100 candidates.
- Returning 30 chunks instead of about 10 answers more questions in every technique. A pool of 100 candidates helps at 30 chunks but hurts the top 10: plain hybrid goes from 113 to 105 answered at 10 chunks.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive. The 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log) and [eval/runs/e31_k30_candidates30.log](eval/runs/e31_k30_candidates30.log).

## Layout

```
src/rag/
├── embed.py          # ONNX embedders: bge-m3 and e5-small (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # SemanticChunker: the semantic cut used by the second stage
│   ├── structure_then_semantic.py  # StructureThenSemanticChunker: headers first, meaning inside big sections
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
