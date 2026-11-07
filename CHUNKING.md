# Chunking technique: semantic

Branch `chunking/semantic`. Cuts where the meaning changes, under a hard token cap. Header structure is used only
as context (each chunk keeps its header path); headers are not cut points here.

## How it works

1. **Units.** The text is split into sentences, bullets and wrapped lines. A header travels with the first unit that
   follows it, so a chunk never starts with an orphan heading.
2. **Windows.** Each unit is embedded together with its neighbours (previous, current, next) using the retrieval model
   (`multilingual-e5-small`, int8 ONNX), so one short sentence does not decide alone.
3. **Cut points.** The distance between consecutive windows is computed; where it is above the 90th percentile of that
   document's distances, the topic shifted and a chunk ends.
4. **Limits.** A chunk is never shorter than `min_tokens` (small segments are merged) and never longer than
   `max_tokens`, measured with the model's own tokenizer (`Embedder.count_tokens`). A single unit above the cap is
   split by `RecursiveCharacterTextSplitter` using that same token count. The cap is our own code; this does not use
   LangChain's `SemanticChunker`.

```python
from rag.ingestion import SemanticChunker, ingest

ingest("data/md", conn, embedder)  # SemanticChunker(embedder) is the default in this branch
ingest("data/md", conn, embedder, chunker=SemanticChunker(embedder, max_tokens=440, min_tokens=100, percentile=90))
```

The embedder must provide `embed(texts, kind)` and `count_tokens(text)`.

## Results (course 1 only, 70 files)

Hit rate / MRR, equal returned text (about 9,000 characters). A+T = tuning sets (40 paraphrased + 10 exact-term),
V = 40 further paraphrased questions used only to validate.

| search | A+T | V |
|---|---|---|
| vector only | 0.800 / 0.559 | 0.725 / 0.558 |
| hybrid (df cap 0.15) | 0.740 / 0.488 | 0.775 / 0.504 |

The best tuning result did not carry over to validation (vector: 0.800 to 0.725).

## Results on the full course (311 files, all 5 courses)

Hit rate / MRR at equal returned text (about 9,000 characters per query). Pooled over 130 questions: 40 + 10 + 40 from
course 1 and 40 from courses 2 to 5. One question is 0.77 points; the standard error of a hit rate near 0.75 is about
3.8 points, so differences under about 4 points are not conclusive.

| technique | chunks | over 512 tokens | vector | hybrid (df cap 0.15) | hybrid (df cap 0.05) |
|---|---|---|---|---|---|
| structural 2400 | 755 | 260 (34%) | 0.662 / 0.489 | 0.708 / 0.542 | 0.731 / 0.565 |
| semantic | 1173 | 1 | 0.700 / 0.465 | 0.731 / 0.496 | 0.738 / 0.521 |
| structure + semantic | 1443 | 0 | 0.723 / 0.500 | 0.754 / 0.493 | **0.762** / 0.562 |

## Boundaries and cost

- Words cut in half: 0. Chunks that continue into the next one inside a sentence: 10 of 248 (4%), caused by the PDF
  conversion (paragraph breaks and page numbers inside sentences), not by the chunker.
- Chunks over the 512-token model limit: 1 of 1173 (515 tokens: a long file path prefix pushed it 3 tokens over the
  440 cap plus margin).
- **Cost:** every unit is embedded in a window, so the first indexing of the full course took about 15 minutes
  against about 2 for the structural chunker. Indexing is incremental by content hash, so the cost is only paid for
  files that change.

## Compared with the other techniques

Smaller chunks (median 268 tokens) that rank the right passage lower than the structural 2400 chunks (MRR 0.521 vs
0.565) with a slightly higher hit rate (0.738 vs 0.731). Full comparison in [eval/results.md](eval/results.md).
Other branches: `chunking/structural`, `chunking/structure-semantic`.
