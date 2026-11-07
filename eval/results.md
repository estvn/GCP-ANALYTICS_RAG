# Retrieval evaluation

Corpus: course 1 of the Google Cloud Data Analytics certificate (70 markdown files, 162 chunks at the default
chunk size of 2400 characters). Embeddings: `multilingual-e5-small`, int8, ONNX. Reproduce the main table with
`uv run python -m rag.evaluation.report`.

Two golden sets, both answered by the passages they cite (a hit is a top-5 chunk that contains the evidence):

- `golden_set.json`: 40 paraphrased questions (22 Spanish, 18 English, 14 of them cross-lingual).
  Drafted by an LLM from the corpus, every evidence snippet verified verbatim against the files.
- `golden_terms.json`: 10 short exact-term queries (acronyms, service names such as `HIPAA`, `IAM`, `Looker`).

## Main result (hit@5 and MRR)

| set | mode | n | hit@5 | MRR |
|---|---|---|---|---|
| golden_set | vector | 40 | 0.650 | 0.528 |
| golden_set | keyword | 40 | 0.525 | 0.385 |
| golden_set | hybrid | 40 | 0.700 | 0.540 |
| golden_terms | vector | 10 | 0.900 | 0.628 |
| golden_terms | keyword | 10 | 1.000 | 0.950 |
| golden_terms | hybrid | 10 | 1.000 | 0.833 |

Hybrid is the robust choice: vectors alone miss exact terms, the keyword leg alone misses paraphrases.

## What the experiments showed

1. **The first hybrid was worse than vectors alone** (hit@5 0.525 vs 0.650). Postgres full-text ranking
   (`ts_rank_cd`) is not BM25: it has no IDF, so words like "para" or "sirve" weigh as much as "BigQuery".
   Dropping query words that appear in more than 15% of the chunks fixed it (0.525 to 0.700). A static
   Spanish/English stoplist was inconsistent and sometimes made MRR worse.
2. **Chunk size, compared at equal returned text** (about 12,000 characters per query, so larger chunks
   do not win just by covering more text): 4000 characters was clearly worst (hit 0.60). 1200 to 2400 were
   within noise of each other. Default stays at 2400 because hybrid scored best there on both sets.
   31% of the 2400-character chunks exceed the embedder's 512-token limit (54% at 4000, 0% at 1200); the
   keyword leg still sees the whole chunk.
3. **Cross-lingual retrieval is the weak point, not ranking.** Within a language hybrid hits 26 of 26
   questions (EN to EN 15/15, ES to ES 11/11). Across languages: ES to EN 1/11, EN to ES 1/3.
4. **Querying in both languages fixes most of it**: fusing the original question with its translation
   gives hit@5 0.925 on the 40 questions (cross-lingual subset 0.14 to 0.79). Implemented as
   `HybridRetriever.search_many`. The translations were written by hand as a proxy for what the calling
   model would send; searching only in the document's language reaches 1.000 but needs knowing that language.
5. **A reranker is not justified.** Even with 30 candidates the answer is present for at most 0.78 to 0.80 of
   the questions (vs 0.70 at 5), and the misses are cross-lingual candidates that never enter the top 30, which a
   reranker cannot recover. Revisit only if a new golden set shows a gap between hit@30 and hit@5.

## Semantic chunking

`SemanticChunker` (`src/rag/ingestion/semantic.py`) splits text into sentences, bullets and header-led units,
embeds each unit with its neighbours, and cuts where the distance between consecutive windows is above a
percentile. It adds a hard cap in real model tokens (440, under the 512-token limit) and a minimum size so
segments are not tiny. The cap is our own code; it uses the model tokenizer, and LangChain's
`RecursiveCharacterTextSplitter` (with a token `length_function`) only splits the rare single unit above the cap.

Same two golden sets, k=5. Ingest times were measured on a loaded machine (the structural default took 14 s when idle).

| chunker | chunks | median / max tokens | over 512 | vector | hybrid | hybrid (terms) |
|---|---|---|---|---|---|---|
| structural 1200 | 268 | 272 / 508 | 0 | 0.700 / 0.481 | 0.550 / 0.383 | 1.0 / 0.733 |
| structural 2400 (default) | 162 | 390 / 983 | 50 | 0.650 / 0.528 | **0.700 / 0.540** | 1.0 / 0.833 |
| semantic p90, min 100 | 257 | 249 / 493 | 0 | 0.700 / 0.548 | 0.625 / 0.414 | 1.0 / 0.750 |
| semantic p75, min 100 | 338 | 186 / 484 | 0 | 0.650 / 0.506 | 0.575 / 0.396 | 1.0 / 0.633 |
| semantic p90, min 200 | 217 | 298 / 493 | 0 | 0.650 / 0.508 | 0.650 / 0.455 | 1.0 / 0.700 |

Cells are hit@5 / MRR on the 40 paraphrased questions unless noted. Compared at equal returned text
(about 9,000 characters per query, since semantic chunks are smaller):

| chunker | mode | paraphrased hit / MRR | terms hit / MRR | chars returned |
|---|---|---|---|---|
| structural 2400, k=5 | hybrid | 0.700 / 0.540 | 1.000 / 0.833 | 9,253 |
| semantic p90 min 100, k=8 | vector only | **0.750** / 0.555 | 1.000 / 0.575 | 8,793 |
| semantic p90 min 100, k=8 | hybrid | 0.675 / 0.422 | 1.000 / 0.750 | 10,491 |

- At the same chunk size semantic cuts are more precise than structural ones (vs structural 1200, about 990
  characters each: hybrid 0.625 vs 0.550, vector MRR 0.548 vs 0.481).
- No chunk exceeds the 512-token limit, unlike 31% of the structural 2400 chunks.
- With semantic chunks the best paraphrase result is vector only; the keyword leg was tuned on structural 2400 and
  adds noise to smaller chunks (hybrid MRR 0.42 vs 0.56). The structural 2400 hybrid remains the best single
  configuration across both sets, so it stays the default.
- Cost: ingestion is 2 to 3 times slower (every unit is embedded, in windows of three).
- Not tried: re-tuning the keyword leg or the RRF weights for semantic chunks. Differences of one or two questions
  are noise on these sets.

## Limits

- Small sets (a single question moves hit@5 by 2.5 points on the main set; ignore differences under about 7 points).
- One correct passage per question: another file that also answers it (for example a summary) counts as a
  miss, so scores are conservative.
- LLM-written questions; paraphrasing was requested, which favours vectors over the keyword leg.
- Only course 1 (70 of the roughly 100 files). int8 vs fp32 embeddings and larger models were not compared.
