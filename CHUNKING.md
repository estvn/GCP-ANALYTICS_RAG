# Chunking technique: semantic + header cuts

Branch `chunking/structure-semantic-e5-small`. Semantic chunking with one extra cut point: every header. It is one pass
over the sentences, not two stages (the two-stage recipe is in `chunking/structure-then-semantic-*`) and it is not the
user's hybrid (that is `chunking/structure-semantic-merge-*`).

## How it works

Same pipeline as `chunking/semantic` (units, 3-unit windows embedded with `multilingual-e5-small`, cut where the
distance between consecutive windows exceeds the 90th percentile, `min_tokens` 100, hard cap of 440 real tokens), plus:

- **Every header is also a cut point** (always on in this branch; it is what `main` does too), provided the current
  chunk already has `min_tokens`. A tiny section is therefore merged into the next one instead of standing alone.

```python
from rag.ingestion import SemanticChunker, ingest

ingest("data/md", conn, embedder)  # SemanticChunker(embedder) by default
```

## Results (course 1 only, 70 files)

Hit rate / MRR, equal returned text (about 9,000 characters). A+T = tuning sets, V = validation set.

| search | A+T | V |
|---|---|---|
| vector only | 0.780 / 0.525 | 0.750 / 0.535 |
| hybrid (df cap 0.15) | 0.760 / 0.495 | 0.775 / 0.505 |

## Results on the full course (311 files, all 5 courses)

Hit rate / MRR at equal returned text (about 9,000 characters per query). Pooled over 130 questions: 40 + 10 + 40 from
course 1 and 40 from courses 2 to 5. One question is 0.77 points; the standard error of a hit rate near 0.75 is about
3.8 points, so differences under about 8 points (two standard errors, about 10 of the 130 questions) are not conclusive.

| technique | chunks | over 512 tokens | vector | hybrid (df cap 0.15) | hybrid (df cap 0.05) |
|---|---|---|---|---|---|
| structural 2400 | 755 | 260 (34%) | 0.662 / 0.489 | 0.708 / 0.542 | 0.731 / 0.565 |
| semantic | 1173 | 1 | 0.700 / 0.465 | 0.731 / 0.496 | 0.738 / 0.521 |
| structure + semantic | 1443 | 0 | 0.723 / 0.500 | 0.754 / 0.493 | **0.762** / 0.562 |

## Boundaries and cost

- Words cut in half: 0. No chunk exceeds the 512-token limit (0 of 1443).
- Smallest chunks of the three techniques (median 217 tokens, 1443 chunks).
- **Cost:** same as `chunking/semantic` (embedding every window). The full-course indexing time we measured for this
  technique (115 s) is not representative: it reused the window embeddings already computed for the semantic run.

## Compared with the other techniques

Best pooled hit rate on the full course (0.762 with the lower document-frequency cap) with a ranking quality
equal to the structural chunker (MRR 0.562 vs 0.565), at the price of a slower first indexing. The gap to the structural
chunker is about 3 points, within the noise of 130 questions. Full comparison in [eval/results.md](eval/results.md).
Other branches: `chunking/structural-e5-small`, `chunking/semantic-e5-small`, `chunking/structure-semantic-e5-small` (and the `-bge-m3` versions).

## Common-method comparison (E29)

Index `full_SEMH` (1408 chunks, avg 883 chars, k=10), embedding model `e5-small`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 94 | 0.723 | 0.500 |
| **hybrid (df cap 0.05)** | **99** | **0.762** | **0.557** |

Hybrid by question set (hit/MRR): A 0.725/0.486 | T 1.000/0.528 | V 0.650/0.480 | F 0.850/0.710. Same-language questions (108): 0.870. Cross-lingual questions (22): 0.227.

Raw lines: [eval/runs/structure-semantic-e5-small.log](eval/runs/structure-semantic-e5-small.log). Table of all techniques (hybrid, hits / MRR):

| technique | e5-small hits / MRR | bge-m3 hits / MRR |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | **114** / .580 |
| community two-stage | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge) | 95 / .552 | 109 / .592 |

The chunk count here (from the E29 database) differs from the one in the sections above and the cause was not verified.
