# Hybrid search with ES/EN RRF fusion (vector + keyword, question and translation)

Branch `retrieval/hybrid-es-en-rrf`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics
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
4. Return the top `k`. The repository implements the 1:1 case as `HybridRetriever.search_many`; the 2:1 weights
   were evaluated with a scratch script.

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E13**: Querying in both languages
- **E25**: e5-small on the full index: same-language vs cross-lingual, and ES+EN fusion
- **E28**: Weighted RRF over two phrasings
- **E30**: Retrieval techniques on the three best chunking indexes

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
- Differences of 2 to 4 questions out of 130 are below the standard error (about 4 points): not conclusive.

## Limits

- Single run, 130 questions. The second phrasing is a hand translation (`eval/golden_translations.json`); the cost of
  producing it in a real system is not measured.
- The evaluation used a keyword document-frequency cap of 0.05 and the weights above from scratch scripts (not in the
  repo). The repo defaults are `MAX_DOC_FREQUENCY = 0.15` and an unweighted `rrf`.
- The user's hybrid chunking and the e5-small indexes were not run through these modes.

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
