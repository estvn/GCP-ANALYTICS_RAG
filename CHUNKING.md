# Chunking technique: semantic only

Branch `chunking/semantic-bge-m3`, embedding model `bge-m3`. Experiment E29 in [eval/experiments.md](eval/experiments.md); all
techniques side by side in [eval/results.md](eval/results.md).

## How it works

1. Split the text into units (sentences, bullets, wrapped lines); a header travels with the first unit after it.
2. Embed each unit together with its neighbours (previous, current, next) with the retrieval model.
3. Cut where the distance between consecutive windows is above the 90th percentile of the document's distances.
4. A chunk is never below 100 tokens (small ones are merged) and never above 440 tokens. Headers are context only,
   not cut points.

## Settings of this run

- Chunker: `SemanticChunker(embedder, 440, 100, 90)` (maximum 440 tokens, minimum 100 tokens where they apply).
- Embedding model: `bge-m3` (int8 ONNX, local) for both cutting (where the chunker uses embeddings) and retrieval.
- Corpus: `data/md`, 310 files (309 with chunks), 5 courses. Database `full_SEM_bge`.
- Cost: indexing resumed from 69 documents: 241 added in 1240 s (about 5 s per document, bge-m3).

```python
from rag.ingestion.semantic import SemanticChunker
from rag.embed import load_embedder
from rag.ingestion.pipeline import ingest

embedder = load_embedder("bge-m3")
ingest("data/md", conn, embedder, chunker=SemanticChunker(embedder, 440, 100, 90))
```

Indexing is incremental by content hash and the chunker settings are part of the hash, so the same call resumes an
interrupted run and a different setting re-indexes.

## Results (E29)

Index `full_SEM_bge` (1128 chunks, avg 1058 chars, k=9), embedding model `bge-m3`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 107 | 0.823 | 0.581 |
| **hybrid (df cap 0.05)** | **113** | **0.869** | **0.576** |

Hybrid by question set (hit/MRR): A 0.850/0.523 | T 0.900/0.587 | V 0.825/0.534 | F 0.925/0.669. Same-language questions (108): 0.917. Cross-lingual questions (22): 0.636.

Raw lines: [eval/runs/semantic-bge-m3.log](eval/runs/semantic-bge-m3.log). The runner script is a session scratch file, **not in the
repository** (see the reproducibility map in `eval/experiments.md`).

## Compared with the other techniques (hybrid, hits / MRR)

| technique | e5-small hits / MRR | bge-m3 hits / MRR |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | **114** / .580 |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

Differences under about 8 points (about 10 of the 130 questions) are not conclusive.

## Limits

- Single run per index, 130 questions. `k` is set by the 9,000-character budget, so techniques with larger chunks return
  fewer results (k from 6 to 10).
- The chunk counts of the e5 indexes of the three older techniques differ from the numbers in older documents and the
  cause was not verified.
- Course 1 questions need the `01_INTRODUCTION_GC_DATA_ANALYTICS/` prefix when loaded with `load_golden`.
