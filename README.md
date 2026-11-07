# Hybrid search with ES/EN RRF fusion (vector + keyword, question and translation)

Branch `retrieval/hybrid-es-en-rrf`. Retrieval over the Google Cloud Data Analytics
course, built to run on a laptop within a small CPU/RAM budget (no GPU). Results use the `bge-m3` embedding model
(int8, ONNX Runtime) and the three most precise chunking techniques (E30).

Raw evaluation output: [eval/runs/e30_retrieval_modes.log](eval/runs/e30_retrieval_modes.log).

## Technique applied

RRF (Reciprocal Rank Fusion) is not a search technique on its own: it only fuses ranked lists. Here it fuses the lists
obtained with the question and with its translation, because embedding models match much better inside one language.

1. Run the hybrid search (vector + keyword, 30 candidates each, fused with RRF) with the question as written.
2. Run the same hybrid search with the translation of the question (Spanish to English or the reverse).
3. Fuse the two result lists with RRF, weights `original:translation`: **1:1** (equal) or **2:1** (the original
   counts double). Score of a chunk = `weight / (60 + position)` in each list, added.
4. Return the top `k`. Both weightings are implemented in this branch: `rrf(..., weights=...)` and
   `HybridRetriever.search_many(..., weights=...)`.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E13**: Querying in both languages
- **E25**: e5-small on the full index: same-language vs cross-lingual, and ES+EN fusion
- **E28**: Weighted RRF over two phrasings
- **E30**: Retrieval techniques on the three best chunking indexes
- **E31**: Returning 30 chunks and a larger candidate pool
- **E32**: Questions nobody used to choose a setting (50 new questions)

## Results (E30, bge-m3, 130 questions: 108 same-language and 22 cross-lingual)

`SEMH` = semantic + header cuts, `SEM` = semantic, `STSEM` = community two-stage. MRR is the mean of 1 / position of the
first correct chunk (1 = always first, 0 if absent).

| chunking | search | answered (130) | MRR | same-language (108) | cross-lingual (22) |
|---|---|---|---|---|---|
| semantic + header cuts | no translation | 114 | 0.580 | 98 | 16 |
| semantic + header cuts | ES/EN 1:1 | 122 | 0.591 | 101 | 21 |
| semantic + header cuts | ES/EN 2:1 | 121 | 0.607 | 102 | 19 |
| semantic | no translation | 113 | 0.576 | 99 | 14 |
| semantic | ES/EN 1:1 | 110 | 0.551 | 91 | 19 |
| semantic | ES/EN 2:1 | 111 | 0.591 | 92 | 19 |
| community two-stage | no translation | 113 | 0.584 | 100 | 13 |
| community two-stage | ES/EN 1:1 | 114 | 0.576 | 95 | 19 |
| community two-stage | ES/EN 2:1 | 112 | 0.605 | 95 | 17 |

## What happened

- Best combination in questions answered: 1:1 on `SEMH` answers 122 of 130 (.938, MRR .591); 2:1 answers 121 (.931, MRR .607). They are tied (1 question).
- It lifts the cross-lingual questions of `SEMH` from .727 (plain hybrid) to .955 (1:1) and .864 (2:1).
- It helps only `SEMH`: on `SEM` it goes from 113 to 110 / 111 and on `STSEM` from 113 to 114 / 112.
- Needs the question translated: a real system pays one extra call and two searches per query.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive.

## Limits

- Single run, 130 questions. The second phrasing is a hand translation (`eval/golden_translations.json`); the cost of
  producing it in a real system is not measured.
- The scripts that produced the E30 tables are not in the repository; the report of this branch (`--translations`)
  evaluates the same modes with the translations in `eval/golden_translations.json`.
- The keyword leg ignores words found in more than 5% of the chunks (`MAX_DOC_FREQUENCY = 0.05` in `store.py`),
  the value used in the evaluation.
- The user's hybrid chunking and the e5-small indexes were not run through these modes.

## Results returning 30 chunks (E31)

Same 130 questions; each query returns up to 30 chunks and the candidate pool is 30 or 100 per search. Answered out of 130; rows are the technique of this branch on the five bge-m3 indexes (the e5-small indexes are in the raw output):

| chunking technique (bge-m3 index) | retrieval | @10 (30 cand.) | @20 (100 cand.) | @30 (30 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| semantic + header cuts | hybrid + ES/EN 1:1 | 122 | 127 | 128 | 130 | 0.553 |
| semantic + header cuts | hybrid + ES/EN 2:1 | 121 | 127 | 127 | 129 | 0.601 |
| structural | hybrid + ES/EN 1:1 | 115 | 127 | 126 | 127 | 0.557 |
| structural | hybrid + ES/EN 2:1 | 116 | 125 | 125 | 129 | 0.579 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 1:1 | 117 | 126 | 126 | 127 | 0.524 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 2:1 | 116 | 126 | 124 | 129 | 0.589 |
| semantic only | hybrid + ES/EN 1:1 | 113 | 121 | 124 | 124 | 0.524 |
| semantic only | hybrid + ES/EN 2:1 | 113 | 125 | 124 | 125 | 0.574 |
| community two-stage | hybrid + ES/EN 1:1 | 114 | 122 | 123 | 125 | 0.540 |
| community two-stage | hybrid + ES/EN 2:1 | 112 | 123 | 123 | 125 | 0.591 |

- Best row: hybrid + ES/EN 1:1 on the semantic + header cuts index, 130 of 130 answered (MRR 0.553); still missing: [].
- Returning 30 chunks answers more questions than about 10; a pool of 100 candidates helps at 30 chunks but can hurt the top 10 and the MRR.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive; the 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log).

## Results on 50 new questions (E32)

50 questions about files that no earlier question used (written by a language model, evidence checked by script), so none of them chose a setting; 30 chunks returned, pool of 30 or 100 candidates. Answered out of 50; rows are the technique of this branch on the five bge-m3 indexes (the e5-small indexes are in the raw output):

| chunking technique (bge-m3 index) | retrieval | @10 (30 cand.) | @30 (30 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|
| semantic + header cuts | hybrid + ES/EN 1:1 | 41 | 48 | 46 | 0.583 |
| semantic + header cuts | hybrid + ES/EN 2:1 | 42 | 46 | 46 | 0.544 |
| structural | hybrid + ES/EN 1:1 | 45 | 50 | 49 | 0.553 |
| structural | hybrid + ES/EN 2:1 | 45 | 49 | 49 | 0.574 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 1:1 | 43 | 48 | 47 | 0.534 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 2:1 | 43 | 46 | 48 | 0.560 |
| semantic only | hybrid + ES/EN 1:1 | 39 | 46 | 43 | 0.513 |
| semantic only | hybrid + ES/EN 2:1 | 39 | 45 | 44 | 0.529 |
| community two-stage | hybrid + ES/EN 1:1 | 42 | 48 | 47 | 0.529 |
| community two-stage | hybrid + ES/EN 2:1 | 43 | 47 | 48 | 0.566 |

- Best row with 100 candidates: hybrid + ES/EN 2:1 on the structural index, 49 of 50 answered (MRR 0.574); still missing: ['h35'].
- With 50 questions one question is 2 points; differences under about 6 questions are not conclusive. Raw output: [eval/runs/e32_heldout_candidates100.log](eval/runs/e32_heldout_candidates100.log).

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
│   ├── fusion.py     # Reciprocal Rank Fusion, optionally weighted per list
│   └── search.py     # HybridRetriever: vector + keyword legs; search_many fuses the question and its translation
└── evaluation/       # golden sets, hit@k / MRR, hybrid report with ES/EN translation (1:1, 2:1)
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
