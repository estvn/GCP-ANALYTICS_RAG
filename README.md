# Vector search (nearest chunks by embedding)

Branch `retrieval/vector`. Hybrid retrieval (Postgres full-text + vectors) over the Google Cloud Data Analytics course,
built to run on a laptop within a small CPU/RAM budget (no GPU). Postgres + pgvector for storage. Results below use the
`bge-m3` embedding model (int8, ONNX Runtime) and the three most precise chunking techniques.

Raw evaluation output: [eval/runs/e30_retrieval_modes.log](eval/runs/e30_retrieval_modes.log).

## Technique applied

1. Embed the question with the same model that embedded the chunks (`bge-m3`; e5 needs the `query: ` prefix).
2. Return the chunks whose embedding is nearest by cosine distance (`store.vector_search`: pgvector `halfvec`,
   HNSW index `m = 16, ef_construction = 64`, `ef_search = 100`).
3. It matches meaning, not words, so it works across languages with a multilingual model.

- **Strong at:** Finds paraphrases and cross-language questions; best MRR of the three techniques.
- **Weak at:** Can miss exact terms (a product name, an error code) that a word search catches.

## ES/EN RRF: the same technique, asked twice

RRF is not a search technique on its own: it only fuses ranked lists. The ES/EN variant runs this technique twice, with
the question as written (original) and with its translation, and fuses the two lists with weights `original:translation`:

- **1:1**: both lists count the same. Score of a chunk = `1 / (60 + position)` in each list, added.
- **2:1**: the original list counts double: `2 / (60 + position_original) + 1 / (60 + position_translation)`.

It helps cross-language questions, because models match much better inside one language; the cost is a translation step
and two searches per query (the evaluation used hand translations from `eval/golden_translations.json`, so that cost is
not measured).

## Experiments

Written up in [eval/experiments.md](https://github.com/estvn/GCP-ANALYTICS_RAG/blob/main/eval/experiments.md) (the file on `main` is the complete log):

- **E05**: Retrieval smoke test (hybrid v1) on the real index
- **E09**: Modes on sets A and T with the final implementation
- **E12**: Language-pair breakdown
- **E13**: Querying in both languages
- **E23**: Same model for index and query?
- **E25**: e5-small on the full index: same-language vs cross-lingual, and ES+EN fusion
- **E26**: bge-m3 on the full index, structural chunking
- **E27**: bge-m3 with structure + semantic chunking
- **E28**: Weighted RRF over two phrasings
- **E30**: Retrieval techniques on the three best chunking indexes

## Results (E30: questions answered out of 130 / MRR, bge-m3)

Rows of this technique in bold. `SEMH` = semantic + header cuts, `SEM` = semantic, `STSEM` = community two-stage.

| retrieval | semantic + headers (`SEMH`) | semantic (`SEM`) | community two-stage (`STSEM`) |
|---|---|---|---|
| vector | **112 / 0.631** | **107 / 0.581** | **107 / 0.595** |
| vector + ES/EN 1:1 | **116 / 0.639** | **104 / 0.548** | **105 / 0.594** |
| vector + ES/EN 2:1 | **116 / 0.638** | **105 / 0.564** | **106 / 0.588** |
| keyword | 75 / 0.355 | 77 / 0.374 | 80 / 0.351 |
| keyword + ES/EN 1:1 | 75 / 0.300 | 79 / 0.313 | 73 / 0.319 |
| keyword + ES/EN 2:1 | 73 / 0.296 | 75 / 0.312 | 76 / 0.320 |
| hybrid | 114 / 0.580 | 113 / 0.576 | 113 / 0.584 |
| hybrid + ES/EN 1:1 | 122 / 0.591 | 110 / 0.551 | 114 / 0.576 |
| hybrid + ES/EN 2:1 | 121 / 0.607 | 111 / 0.591 | 112 / 0.605 |

`k` is the number of chunks returned per query (10, 9 and 10 here: a 9,000-character budget divided by the average
chunk length). MRR is the mean of 1 / position of the first correct chunk (1 = always first, 0 if absent).

## What happened

- Best MRR of all combinations: vector + ES/EN on `SEMH` (116 answered, MRR 0.639); without translation vector on `SEMH` answers 112 (MRR .631).
- Vector beats the hybrid in MRR in every chunking index (`SEMH` .631 against .580).
- Translation helps `SEMH` (112 to 116) and is neutral or slightly worse on `SEM` (107 to 104/105) and `STSEM` (107 to 105/106).
- Differences of 2 to 4 questions out of 130 are below the standard error (about 4 points): not conclusive.

## Limits

- Single run, 130 questions. The second phrasing is a hand translation; a real system needs a translation step.
- The evaluation used a keyword document-frequency cap of 0.05 and ES/EN weights of 2:1 from scratch scripts (not in the
  repo). The repo defaults are `MAX_DOC_FREQUENCY = 0.15` in `store.py` and an unweighted `rrf`.
- The user's hybrid chunking and the e5-small indexes were not run through these retrieval modes.

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
