# Chunking technique: structural (headers, 2400 characters)

Branch `chunking/structural-bge-m3`, embedding model `bge-m3`. Experiment E29 in [eval/experiments.md](eval/experiments.md); all
techniques side by side in [eval/results.md](eval/results.md).

## How it works

1. Split by markdown headers (`MarkdownHeaderTextSplitter`).
2. Pack neighbouring pieces and split what is too long with `RecursiveCharacterTextSplitter`, at most 2400
   **characters** with an overlap of 200 (it is characters, not tokens; see E19 and E22).
3. No embedding is used to cut, so it is the cheapest technique to index.

## Settings of this run

- Chunker: `StructuralChunker(2400, 200)` (maximum 440 tokens, minimum 100 tokens where they apply).
- Embedding model: `bge-m3` (int8 ONNX, local) for both cutting (where the chunker uses embeddings) and retrieval.
- Corpus: `data/md`, 310 files (309 with chunks), 5 courses. Database `full_S2400_bge`.
- Cost: indexing time of this run was not recorded.

```python
from rag.ingestion.chunk import StructuralChunker
from rag.embed import load_embedder
from rag.ingestion.pipeline import ingest

embedder = load_embedder("bge-m3")
ingest("data/md", conn, embedder, chunker=StructuralChunker(2400, 200))
```

Indexing is incremental by content hash and the chunker settings are part of the hash, so the same call resumes an
interrupted run and a different setting re-indexes.

## Results (E29)

Index `full_S2400_bge` (741 chunks, avg 1567 chars, k=6), embedding model `bge-m3`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 102 | 0.785 | 0.603 |
| **hybrid (df cap 0.05)** | **107** | **0.823** | **0.612** |

Hybrid by question set (hit/MRR): A 0.800/0.556 | T 0.900/0.667 | V 0.750/0.534 | F 0.900/0.731. Same-language questions (108): 0.898. Cross-lingual questions (22): 0.455.

Raw lines: [eval/runs/structural-bge-m3.log](eval/runs/structural-bge-m3.log). The runner script is a session scratch file, **not in the
repository** (see the reproducibility map in `eval/experiments.md`).

## Compared with the other techniques (hybrid, hits / MRR)

| technique | e5-small hits / MRR | bge-m3 hits / MRR |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | **114** / .580 |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

A difference of 2 to 4 questions out of 130 is below the standard error (about 4 points): it is not conclusive.

## Limits

- Single run per index, 130 questions. `k` is set by the 9,000-character budget, so techniques with larger chunks return
  fewer results (k from 6 to 10).
- The chunk counts of the e5 indexes of the three older techniques differ from the numbers in older documents and the
  cause was not verified.
- Course 1 questions need the `01_INTRODUCTION_GC_DATA_ANALYTICS/` prefix when loaded with `load_golden`.
