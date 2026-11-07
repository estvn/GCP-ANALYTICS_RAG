# Chunking technique: user's hybrid (community two-stage + merge of neighbouring pieces)

Branch `chunking/structure-semantic-merge-e5-small`, embedding model `e5-small`. Experiment E29 in [eval/experiments.md](eval/experiments.md); all
techniques side by side in [eval/results.md](eval/results.md).

## How it works

1. **Structural stage:** split by markdown headers.
2. **Semantic stage:** a section above 440 tokens is subdivided by the semantic cut chunker.
3. **Merge step:** embed every piece and repeatedly merge the most similar pair of neighbours in the document. A pair
   merges when its similarity reaches the threshold (a percentile, here 25, of the document's own neighbour
   similarities) or when one side is below 100 tokens, and never past 440 tokens. Merging may cross a section
   boundary; the merged chunk keeps the headers of its first piece. Nothing is discarded.

## Settings of this run

- Chunker: `StructureSemanticMergeChunker(embedder, 440, 100, 25)` (maximum 440 tokens, minimum 100 tokens where they apply).
- Embedding model: `e5-small` (int8 ONNX, local) for both cutting (where the chunker uses embeddings) and retrieval.
- Corpus: `data/md`, 310 files (309 with chunks), 5 courses. Database `full_STMERGE_e5`.
- Cost: full indexing of 310 files in 185 s (e5-small).

```python
from rag.ingestion.structure_semantic_merge import StructureSemanticMergeChunker
from rag.embed import load_embedder
from rag.ingestion.pipeline import ingest

embedder = load_embedder("e5-small")
ingest("data/md", conn, embedder, chunker=StructureSemanticMergeChunker(embedder, 440, 100, 25))
```

Indexing is incremental by content hash and the chunker settings are part of the hash, so the same call resumes an
interrupted run and a different setting re-indexes.

## Results (E29)

Index `full_STMERGE_e5` (907 chunks, avg 1267 chars, k=7), embedding model `e5-small`, 130 questions, about 9,000 characters returned per query.

| search | hits / 130 | hit rate | MRR |
|---|---|---|---|
| vector only | 89 | 0.685 | 0.447 |
| **hybrid (df cap 0.05)** | **95** | **0.731** | **0.552** |

Hybrid by question set (hit/MRR): A 0.650/0.477 | T 0.900/0.483 | V 0.675/0.498 | F 0.825/0.698. Same-language questions (108): 0.861. Cross-lingual questions (22): 0.091.

Raw lines: [eval/runs/structure-semantic-merge-e5-small.log](eval/runs/structure-semantic-merge-e5-small.log). The runner script is a session scratch file, **not in the
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
