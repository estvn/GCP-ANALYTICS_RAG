# Chunking technique: semantic + header cuts (called `structure-semantic` in the branch name; NOT the user's hybrid)

Branch `chunking/structure-semantic-bge-m3`, embedding model `bge-m3`. Experiment E29 in [eval/experiments.md](eval/experiments.md); all
techniques side by side in [eval/results.md](eval/results.md).

## How it works

Same as the semantic chunker, with one extra cut point:
1. Units and window embeddings as in the semantic chunker.
2. A chunk is cut at a topic shift (90th percentile) or at the 440-token cap, **and also at every `#` header once
   the current chunk already has 100 tokens** (`starts_section and tokens >= min_tokens`).
3. It is one pass over sentences, with no second stage and no merge. The extra cut explains why it makes more and
   smaller chunks than the pure semantic chunker.

## Settings of this run

- Chunker: `SemanticChunker(embedder, 440, 100, 90)` (maximum 440 tokens, minimum 100 tokens where they apply).
- Embedding model: `bge-m3` (int8 ONNX, local) for both cutting (where the chunker uses embeddings) and retrieval.
- Corpus: `data/md`, 310 files (309 with chunks), 5 courses. Database `full_SEMH_bge`.
- Cost: indexing resumed after an interruption (E27): 132 documents in 1194 s (bge-m3).

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

Index `full_SEMH_bge` (1416 chunks, avg 879 chars, k=10), embedding model `bge-m3`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 112 | 0.862 | 0.631 |
| **hybrid (df cap 0.05)** | **114** | **0.877** | **0.580** |

Hybrid by question set (hit/MRR): A 0.925/0.564 | T 1.000/0.510 | V 0.775/0.527 | F 0.900/0.666. Same-language questions (108): 0.907. Cross-lingual questions (22): 0.727.

Raw lines: [eval/runs/structure-semantic-bge-m3.log](eval/runs/structure-semantic-bge-m3.log). The runner script is a session scratch file, **not in the
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
