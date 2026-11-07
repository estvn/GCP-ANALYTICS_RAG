# Semantic chunking with header cuts

Branch `chunking/structure-semantic-bge-m3`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Embedding model of
this branch's results: `bge-m3` (int8, ONNX Runtime).

Full description, settings and how to reproduce: [CHUNKING.md](CHUNKING.md). Raw evaluation lines:
[eval/runs/structure-semantic-bge-m3.log](eval/runs/structure-semantic-bge-m3.log).

## Technique applied

Same as the semantic chunker, with one extra cut point:
1. Units and window embeddings as in the semantic chunker.
2. A chunk is cut at a topic shift (90th percentile) or at the 440-token cap, **and also at every `#` header once
   the current chunk already has 100 tokens** (`starts_section and tokens >= min_tokens`).
3. It is one pass over sentences, with no second stage and no merge. The extra cut explains why it makes more and
   smaller chunks than the pure semantic chunker.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E17**: Grid with a validation set: 6 chunkers x 7 search variants
- **E19**: Full-index evaluation of the three chunkers
- **E21**: How structure and meaning share the cuts
- **E27**: bge-m3 with structure + semantic chunking
- **E29**: All chunking techniques on one common method
- **E31**: Returning 30 chunks and a larger candidate pool
- **E32**: Questions nobody used to choose a setting (50 new questions)

## Results (E29: hybrid search, 130 questions, hits / MRR)

| technique | e5-small | bge-m3 |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | 114 / .580 **◀ this branch** |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

Hits are questions answered out of 130, with about 9,000 characters returned per query. Differences under about 8 points
(about 10 of the 130 questions) are not conclusive.

## Results returning 30 chunks (E31)

Index `full_SEMH_bge` (`bge-m3`), the same 130 questions; each query returns up to 30 chunks (about 26,356 characters). Questions answered out of 130 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search (the pool of chunks each search contributes before fusion):

| retrieval technique | @10 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| vector | 112 | 125 | 112 | 122 | 125 | 0.637 |
| vector + ES/EN 1:1 | 116 | 126 | 117 | 123 | 127 | 0.645 |
| vector + ES/EN 2:1 | 116 | 125 | 115 | 124 | 127 | 0.644 |
| keyword | 75 | 94 | 75 | 88 | 94 | 0.365 |
| keyword + ES/EN 1:1 | 75 | 102 | 73 | 89 | 102 | 0.297 |
| keyword + ES/EN 2:1 | 73 | 96 | 74 | 91 | 98 | 0.312 |
| hybrid | 114 | 127 | 105 | 121 | 126 | 0.580 |
| hybrid + ES/EN 1:1 | 122 | 128 | 119 | 127 | 130 | 0.553 |
| hybrid + ES/EN 2:1 | 121 | 127 | 122 | 127 | 129 | 0.601 |

- Best technique on this index: hybrid + ES/EN 1:1, 130 of 130 answered with 100 candidates (MRR 0.553); still missing: [].
- At 20 chunks (about 17,571 characters) the best technique answers 127 of 130 with 100 candidates.
- Returning 30 chunks instead of about 10 answers more questions in every technique. A pool of 100 candidates helps at 30 chunks but hurts the top 10: plain hybrid goes from 114 to 105 answered at 10 chunks.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive. The 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log) and [eval/runs/e31_k30_candidates30.log](eval/runs/e31_k30_candidates30.log).

## Results on 50 new questions (E32)

Index `full_SEMH_bge` (`bge-m3`). 50 questions about files that no earlier question used (written by a language model, evidence checked by script), so none of them chose a setting. Questions answered out of 50 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search:

| retrieval technique | @10 (30 cand.) | @20 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|---|
| vector | 39 | 42 | 44 | 39 | 42 | 44 | 0.534 |
| vector + ES/EN 1:1 | 37 | 44 | 44 | 37 | 43 | 44 | 0.505 |
| vector + ES/EN 2:1 | 37 | 44 | 44 | 37 | 44 | 44 | 0.527 |
| keyword | 32 | 36 | 38 | 32 | 36 | 38 | 0.471 |
| keyword + ES/EN 1:1 | 41 | 42 | 44 | 36 | 42 | 44 | 0.444 |
| keyword + ES/EN 2:1 | 34 | 37 | 39 | 33 | 36 | 39 | 0.460 |
| hybrid | 39 | 46 | 46 | 38 | 41 | 46 | 0.516 |
| hybrid + ES/EN 1:1 | 41 | 46 | 48 | 42 | 44 | 46 | 0.583 |
| hybrid + ES/EN 2:1 | 42 | 46 | 46 | 41 | 45 | 46 | 0.544 |

- Best technique on this index: hybrid + ES/EN 1:1, 46 of 50 answered with 100 candidates (MRR 0.583); still missing: ['h24', 'h25', 'h30', 'h37'].
- With 50 questions one question is 2 points; differences under about 6 questions are not conclusive. A pool of 100 candidates is not better than 30 here. Raw output: [eval/runs/e32_heldout_candidates100.log](eval/runs/e32_heldout_candidates100.log) and [eval/runs/e32_heldout_candidates30.log](eval/runs/e32_heldout_candidates30.log).

## Layout

```
src/rag/
├── embed.py          # ONNX embedders: bge-m3 and e5-small (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # SemanticChunker: semantic cuts + header cuts
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
