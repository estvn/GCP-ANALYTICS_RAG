# Keyword search (by words, Postgres full-text)

Branch `retrieval/keyword`. Retrieval over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Results below use the
`bge-m3` embedding model (int8, ONNX Runtime) and the three most precise chunking techniques.

Raw evaluation output: [eval/runs/e30_retrieval_modes.log](eval/runs/e30_retrieval_modes.log).

## Technique applied

1. Take the words of the question with 3 or more letters, lower-cased.
2. Drop words found in too many chunks (document-frequency cap; Postgres ranking has no IDF, so this approximates it).
3. Match chunks that contain any remaining word (`tsvector`, `simple` config, GIN index) and rank them with
   `ts_rank_cd` (`store.keyword_search`). It is **not BM25**.

- **Strong at:** Catches exact terms and names that embeddings blur.
- **Weak at:** No meaning and no cross-language: a Spanish question finds almost nothing in English chunks.

## ES/EN RRF: the same technique, asked twice

RRF is not a search technique on its own: it only fuses ranked lists. The ES/EN variant runs this technique twice, with
the question as written (original) and with its translation, and fuses the two lists with weights `original:translation`:

- **1:1**: both lists count the same. Score of a chunk = `1 / (60 + position)` in each list, added.
- **2:1**: the original list counts double: `2 / (60 + position_original) + 1 / (60 + position_translation)`.

It helps cross-language questions, because models match much better inside one language; the cost is a translation step
and two searches per query (the evaluation used hand translations from `eval/golden_translations.json`, so that cost is
not measured).

This branch holds the single-phrasing technique. There is no branch for keyword + ES/EN: it did not improve the total
answered, so its code was not kept.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E06**: First baseline matrix: 3 chunk sizes x 3 modes
- **E07**: Fixing the keyword leg: stoplist, document-frequency cap, RRF weights
- **E09**: Modes on sets A and T with the final implementation
- **E17**: Grid with a validation set: 6 chunkers x 7 search variants
- **E25**: e5-small on the full index: same-language vs cross-lingual, and ES+EN fusion
- **E28**: Weighted RRF over two phrasings
- **E30**: Retrieval techniques on the three best chunking indexes
- **E31**: Returning 30 chunks and a larger candidate pool

## Results (E30: questions answered out of 130 / MRR, bge-m3)

Rows of this technique in bold. `SEMH` = semantic + header cuts, `SEM` = semantic, `STSEM` = community two-stage.

| retrieval | semantic + headers (`SEMH`) | semantic (`SEM`) | community two-stage (`STSEM`) |
|---|---|---|---|
| vector | 112 / 0.631 | 107 / 0.581 | 107 / 0.595 |
| vector + ES/EN 1:1 | 116 / 0.639 | 104 / 0.548 | 105 / 0.594 |
| vector + ES/EN 2:1 | 116 / 0.638 | 105 / 0.564 | 106 / 0.588 |
| keyword | **75 / 0.355** | **77 / 0.374** | **80 / 0.351** |
| keyword + ES/EN 1:1 | **75 / 0.300** | **79 / 0.313** | **73 / 0.319** |
| keyword + ES/EN 2:1 | **73 / 0.296** | **75 / 0.312** | **76 / 0.320** |
| hybrid | 114 / 0.580 | 113 / 0.576 | 113 / 0.584 |
| hybrid + ES/EN 1:1 | 122 / 0.591 | 110 / 0.551 | 114 / 0.576 |
| hybrid + ES/EN 2:1 | 121 / 0.607 | 111 / 0.591 | 112 / 0.605 |

`k` is the number of chunks returned per query (10, 9 and 10 here: a 9,000-character budget divided by the average
chunk length). MRR is the mean of 1 / position of the first correct chunk (1 = always first, 0 if absent).

## What happened

- The weakest technique alone: 75 to 80 questions answered out of 130 in the three indexes (hit rate .58 to .62).
- Cross-lingual questions: .09 to .14 without translation. With ES/EN 1:1 it reaches .73 to .82 on those 22 questions, but same-language drops (`SEMH` .676 to .528), so the total does not improve.
- Its value is as the second leg of the hybrid, for exact terms.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive.

## Limits

- Single run, 130 questions. The second phrasing is a hand translation; a real system needs a translation step.
- The keyword leg ignores words found in more than 5% of the chunks (`MAX_DOC_FREQUENCY = 0.05` in `store.py`), the value used in the evaluation.
- The user's hybrid chunking and the e5-small indexes were not run through these retrieval modes.

## Results returning 30 chunks (E31)

Same 130 questions; each query returns up to 30 chunks and the candidate pool is 30 or 100 per search. Answered out of 130; rows are the technique of this branch on the five bge-m3 indexes (the e5-small indexes are in the raw output):

| chunking technique (bge-m3 index) | retrieval | @10 (30 cand.) | @20 (100 cand.) | @30 (30 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| semantic + header cuts | keyword | 75 | 88 | 94 | 94 | 0.365 |
| structural | keyword | 81 | 95 | 100 | 100 | 0.395 |
| user's hybrid (two-stage + merge) | keyword | 85 | 95 | 100 | 100 | 0.390 |
| semantic only | keyword | 81 | 91 | 96 | 96 | 0.384 |
| community two-stage | keyword | 80 | 94 | 98 | 98 | 0.359 |

- Best row: keyword on the structural index, 100 of 130 answered (MRR 0.395); still missing: ['q01', 'q03', 'q06', 'q08', 'q10', 'q14', 'q21', 'q23', 'q24', 'q26', 'q30', 'q34', 'q35', 'q40', 'k09', 'v01', 'v02', 'v03', 'v05', 'v06', 'v08', 'v17', 'v19', 'v20', 'v30', 'f12', 'f18', 'f35', 'f36', 'f40'].
- Returning 30 chunks answers more questions than about 10; a pool of 100 candidates helps at 30 chunks but can hurt the top 10 and the MRR.
- Differences under about 8 points (about 10 of the 130 questions) are not conclusive; the 130 questions also chose the settings, so the figures are optimistic for new questions. Raw output: [eval/runs/e31_k30_candidates100.log](eval/runs/e31_k30_candidates100.log).

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
│   └── search.py     # KeywordRetriever: discriminating words ranked by ts_rank_cd
└── evaluation/       # golden sets, hit@k / MRR, keyword report
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
