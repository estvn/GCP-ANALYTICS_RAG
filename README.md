# Hybrid chunking (two-stage + merge of neighbours)

Branch `chunking/structure-semantic-merge-e5-small`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Embedding model of
this branch's results: `e5-small` (int8, ONNX Runtime).

Full description, settings and how to reproduce: [CHUNKING.md](CHUNKING.md). Raw evaluation lines:
[eval/runs/structure-semantic-merge-e5-small.log](eval/runs/structure-semantic-merge-e5-small.log).

## Technique applied

1. **Structural stage:** split by markdown headers.
2. **Semantic stage:** a section above 440 tokens is subdivided by the semantic cut chunker.
3. **Merge step:** embed every piece and repeatedly merge the most similar pair of neighbours in the document. A pair
   merges when its similarity reaches the threshold (a percentile, here 25, of the document's own neighbour
   similarities) or when one side is below 100 tokens, and never past 440 tokens. Merging may cross a section
   boundary; the merged chunk keeps the headers of its first piece. Nothing is discarded.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E29**: All chunking techniques on one common method
- **E31**: Returning 30 chunks and a larger candidate pool
- **E32**: Questions nobody used to choose a setting (50 new questions)

## Results (E29: hybrid search, 130 questions, hits / MRR)

| technique | e5-small | bge-m3 |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | 114 / .580 |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 **◀ this branch** | 109 / .592 |

Hits are questions answered out of 130, with about 9,000 characters returned per query. Differences under about 8 points
(about 10 of the 130 questions) are not conclusive.

## Results returning 30 chunks (E31)

Index `full_STMERGE_e5` (`e5-small`), the same 130 questions; each query returns up to 30 chunks (about 38,009 characters). Questions answered out of 130 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search (the pool of chunks each search contributes before fusion):

| retrieval technique | @10 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| vector | 94 | 108 | 94 | 105 | 108 | 0.458 |
| vector + ES/EN 1:1 | 99 | 117 | 78 | 106 | 112 | 0.271 |
| vector + ES/EN 2:1 | 93 | 108 | 92 | 104 | 110 | 0.338 |
| keyword | 86 | 99 | 86 | 95 | 99 | 0.407 |
| keyword + ES/EN 1:1 | 84 | 110 | 81 | 100 | 108 | 0.317 |
| keyword + ES/EN 2:1 | 86 | 101 | 84 | 98 | 101 | 0.357 |
| hybrid | 102 | 111 | 100 | 102 | 107 | 0.556 |
| hybrid + ES/EN 1:1 | 110 | 122 | 98 | 117 | 121 | 0.387 |
| hybrid + ES/EN 2:1 | 105 | 111 | 105 | 109 | 111 | 0.454 |

- Best technique on this index: hybrid + ES/EN 1:1, 121 of 130 answered with 100 candidates (MRR 0.387); still missing: ['q16', 'q35', 'v02', 'v05', 'v08', 'v24', 'v30', 'v35', 'f12'].
- At 20 chunks (about 25,339 characters) the best technique answers 117 of 130 with 100 candidates.
- Returning 30 chunks instead of about 10 answers more questions in every technique. A pool of 100 candidates helps at 30 chunks but hurts the top 10: plain hybrid goes from 102 to 100 answered at 10 chunks.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive. The 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log) and [eval/runs/e31_k30_candidates30.log](eval/runs/e31_k30_candidates30.log).

## Results on 50 new questions (E32)

Index `full_STMERGE_e5` (`e5-small`). 50 questions about files that no earlier question used (written by a language model, evidence checked by script), so none of them chose a setting. Questions answered out of 50 within the first 10, 20 and 30 chunks, with 30 and with 100 candidates per search:

| retrieval technique | @10 (30 cand.) | @20 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|---|
| vector | 41 | 44 | 45 | 41 | 44 | 45 | 0.624 |
| vector + ES/EN 1:1 | 46 | 49 | 49 | 39 | 46 | 48 | 0.546 |
| vector + ES/EN 2:1 | 42 | 44 | 45 | 41 | 44 | 45 | 0.585 |
| keyword | 32 | 37 | 40 | 32 | 37 | 40 | 0.433 |
| keyword + ES/EN 1:1 | 38 | 42 | 44 | 33 | 42 | 45 | 0.389 |
| keyword + ES/EN 2:1 | 33 | 39 | 41 | 31 | 40 | 41 | 0.382 |
| hybrid | 42 | 45 | 45 | 40 | 43 | 45 | 0.594 |
| hybrid + ES/EN 1:1 | 46 | 49 | 50 | 44 | 47 | 49 | 0.620 |
| hybrid + ES/EN 2:1 | 43 | 45 | 45 | 44 | 46 | 46 | 0.638 |

- Best technique on this index: hybrid + ES/EN 1:1, 49 of 50 answered with 100 candidates (MRR 0.620); still missing: ['h09'].
- With 50 questions one question is 2 points; differences under about 6 questions are not conclusive. A pool of 100 candidates is not better than 30 here. Raw output: [eval/runs/e32_heldout_candidates100.log](eval/runs/e32_heldout_candidates100.log) and [eval/runs/e32_heldout_candidates30.log](eval/runs/e32_heldout_candidates30.log).

## Layout

```
src/rag/
├── embed.py          # ONNX embedders: bge-m3 and e5-small (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # SemanticChunker: the semantic cut used by the second stage
│   ├── structure_semantic_merge.py  # StructureSemanticMergeChunker: two stages + merge of neighbouring pieces
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
