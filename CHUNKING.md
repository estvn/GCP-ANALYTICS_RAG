# Chunking technique: structural

Branch `chunking/structural`. This is the default technique on `main`; this branch keeps only its code.

## How it works

1. **Split by headers.** `MarkdownHeaderTextSplitter` cuts the document at `#`, `##` and `###`. Each section keeps its
   header and its header path (`file > header > subheader`).
2. **Merge small sections.** Adjacent sections of the same file are joined until they reach the size limit. The course
   sections are tiny (median 53 words), so cutting only at headers would produce chunks too small to be useful.
3. **Split oversized sections recursively.** `RecursiveCharacterTextSplitter` tries paragraph, line, then sentence
   ends (`. `, `? `, `! `, `; `) before a plain space, with overlap between pieces. The section header stays attached
   to the first piece so it is never left alone in a chunk.

No model is used, so indexing is fast (the full course, 311 files, took about 2 minutes).

```python
from rag.ingestion import StructuralChunker, ingest

ingest("data/md", conn, embedder, chunker=StructuralChunker(chunk_size=2400, overlap=200))
```

`chunk_size` is in **characters** (about 4 per token). `overlap` defaults to 200.

## Results (course 1, 70 files, equal returned text of about 9,000 characters per query)

Hit rate / MRR, hybrid retrieval (document-frequency cap 0.15, RRF 1:1). A+T = 40 paraphrased + 10 exact-term
questions used to tune; V = 40 further paraphrased questions used only to validate.

| chunk size | chunks | A+T | V |
|---|---|---|---|
| 1800 | 198 | 0.760 / 0.553 | 0.700 / 0.516 |
| **2400 (default)** | 162 | 0.760 / **0.598** | 0.750 / 0.498 |

Smaller (1200) did worse in hybrid on the first set, and 4000 was clearly the worst at equal returned text.

## Boundaries and limits

- Words cut in half: 0 (every chunk on course 1).
- Chunks cut inside a sentence: 7 of 162 (4%), all caused by the PDF conversion inserting a paragraph break or a page
  number in the middle of a sentence, not by the splitter.
- **34% of the chunks (260 of 755 on the full course) exceed the embedder's 512-token limit**, because the size limit
  is in characters. The vector for those chunks is computed from the first 512 tokens only; the keyword search still
  sees the whole chunk.

## Compared with the other techniques

Hit rates tie with the semantic techniques within the noise of the evaluation sets; this one has the best MRR, is
the simplest and the fastest to index. Full comparison: [eval/results.md](eval/results.md). Other branches:
`chunking/semantic`, `chunking/structure-semantic`.
