# Chunking technique: community two-stage (headers first, semantic only inside big sections)

Branch `chunking/structure-then-semantic-e5-small`, embedding model `e5-small`. Experiment E29 in [eval/experiments.md](eval/experiments.md); all
techniques side by side in [eval/results.md](eval/results.md).

## How it works

1. **Structural stage:** split by markdown headers; consecutive sections below 100 tokens are merged, never past 440.
2. **Semantic stage:** only a section above 440 tokens is subdivided by the semantic cut chunker (90th percentile).
3. Small sections are never embedded to be cut, so it is cheaper than the one-pass chunkers. This is the community
   recipe; the user's hybrid is this plus a merge step (see `structure-semantic-merge-*`).

## Settings of this run

- Chunker: `StructureThenSemanticChunker(embedder, 440, 100, 90)` (maximum 440 tokens, minimum 100 tokens where they apply).
- Embedding model: `e5-small` (int8 ONNX, local) for both cutting (where the chunker uses embeddings) and retrieval.
- Corpus: `data/md`, 310 files (309 with chunks), 5 courses. Database `full_STSEM_e5`.
- Cost: full indexing of 310 files in 150 s (e5-small).

```python
from rag.ingestion.structure_then_semantic import StructureThenSemanticChunker
from rag.embed import load_embedder
from rag.ingestion.pipeline import ingest

embedder = load_embedder("e5-small")
ingest("data/md", conn, embedder, chunker=StructureThenSemanticChunker(embedder, 440, 100, 90))
```

Indexing is incremental by content hash and the chunker settings are part of the hash, so the same call resumes an
interrupted run and a different setting re-indexes.

## Results (E29)

Index `full_STSEM_e5` (1335 chunks, avg 911 chars, k=10), embedding model `e5-small`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 91 | 0.700 | 0.476 |
| **hybrid (df cap 0.05)** | **97** | **0.746** | **0.508** |

Hybrid by question set (hit/MRR): A 0.675/0.451 | T 0.900/0.517 | V 0.650/0.458 | F 0.875/0.612. Same-language questions (108): 0.870. Cross-lingual questions (22): 0.136.

Raw lines: [eval/runs/structure-then-semantic-e5-small.log](eval/runs/structure-then-semantic-e5-small.log). The runner script is a session scratch file, **not in the
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
