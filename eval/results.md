# Retrieval evaluation

The full ledger of every experiment (including retracted and pending ones) is in [experiments.md](experiments.md).

Corpus: course 1 of the Google Cloud Data Analytics certificate (70 markdown files, 162 chunks at the default
chunk size of 2400 characters). Embeddings: `multilingual-e5-small`, int8, ONNX. Reproduce the main table with
`uv run python -m rag.evaluation.report --model e5-small` (the report now defaults to `--model bge-m3`).

Three golden sets, all answered by the passages they cite (a hit is a top-5 chunk that contains the evidence):

- `golden_set.json`: 40 paraphrased questions (22 Spanish, 18 English, 14 of them cross-lingual).
  Drafted by an LLM from the corpus, every evidence snippet verified verbatim against the files.
- `golden_terms.json`: 10 short exact-term queries (acronyms, service names such as `HIPAA`, `IAM`, `Looker`).
- `golden_validation.json`: 40 further paraphrased questions (20 ES, 20 EN), drafted separately and **not used to
  choose any parameter**, so it checks whether a choice made on the other two sets holds up.

## Main result (hit@5 and MRR)

| set | mode | n | hit@5 | MRR |
|---|---|---|---|---|
| golden_set | vector | 40 | 0.650 | 0.528 |
| golden_set | keyword | 40 | 0.525 | 0.385 |
| golden_set | hybrid | 40 | 0.700 | 0.540 |
| golden_terms | vector | 10 | 0.900 | 0.628 |
| golden_terms | keyword | 10 | 1.000 | 0.950 |
| golden_terms | hybrid | 10 | 1.000 | 0.833 |
| golden_validation | vector | 40 | 0.700 | 0.574 |
| golden_validation | keyword | 40 | 0.475 | 0.295 |
| golden_validation | hybrid | 40 | 0.675 | 0.486 |

On exact terms the keyword leg matters (vectors alone miss `Google Gemini`). On paraphrased questions hybrid and
vector-only are equivalent: hybrid led on the first set (0.700 vs 0.650) but not on the validation set
(0.675 vs 0.700, MRR 0.486 vs 0.574), so that first advantage was partly tuning noise. Hybrid is kept as the
default because users also type acronyms and service names, not because it wins on paraphrases.

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
  configuration across both sets, so it stayed the default at the time (superseded: see the E29 section; the
  default on main is now semantic with header cuts).
- Cost: ingestion is 2 to 3 times slower (every unit is embedded, in windows of three).
- Not tried: re-tuning the keyword leg or the RRF weights for semantic chunks. Differences of one or two questions
  are noise on these sets.

## Structure first, semantic inside, and a tuning grid

`SemanticChunker(split_on_headers=True)` makes every header a cut point (structure first) and then cuts inside
each section on meaning shifts, under the token cap. Six chunkers were compared with seven search variants each
(vector only; hybrid with a document-frequency cap of 0.10, 0.15 or 0.30 and RRF weights 1:1 or 2:1), at equal
returned text (about 9,000 characters per query, so `k` differs per chunker). Parameters were chosen on the
tuning sets (A+T: 40 paraphrased plus 10 exact-term questions) and then checked on the validation set (V).
Cells are hit / MRR.

| chunker | chunks | k | vector, A+T | vector, V | hybrid (df 0.15, 1:1), A+T | hybrid, V |
|---|---|---|---|---|---|---|
| structural 2400 (default) | 162 | 6 | 0.740 / 0.555 | 0.725 / 0.578 | 0.760 / **0.598** | 0.750 / 0.498 |
| structural 1800 | 198 | 7 | 0.740 / 0.542 | 0.700 / 0.580 | 0.760 / 0.553 | 0.700 / 0.516 |
| semantic p90, min 100 | 257 | 9 | 0.800 / 0.559 | 0.725 / 0.558 | 0.740 / 0.488 | 0.775 / 0.504 |
| structure + semantic p90, min 100 | 310 | 11 | 0.780 / 0.525 | 0.750 / 0.535 | 0.760 / 0.495 | 0.775 / 0.505 |
| structure + semantic p90, min 200 | 231 | 8 | 0.760 / 0.529 | 0.750 / 0.526 | 0.740 / 0.485 | 0.700 / 0.521 |
| structure + semantic p75, min 200 | 252 | 9 | 0.800 / 0.525 | 0.750 / 0.486 | 0.760 / 0.487 | 0.700 / 0.467 |

- **No chunker is clearly better.** Hit rates sit between 0.70 and 0.80 and the gaps are a few questions. The
  best tuning result does not carry over: semantic p90 min 100 with vectors scores 0.800 on A+T and 0.725 on V.
- **Rank quality favours the structural 2400 chunks** (best MRR on the tuning sets with hybrid, and with vectors
  on V): smaller semantic chunks compete with similar neighbours and the right one lands lower.
- **A document-frequency cap of 0.30 is worse than 0.10 or 0.15 for every chunker.** Weighting vectors 2:1 gives no
  consistent gain, so the default (0.15, 1:1) stayed at the time (since changed: 0.05 on main).
- **Semantic and structure-first chunks fit the embedder** (none above 512 tokens, against 31% of the structural
  2400 chunks), but retrieval quality did not improve in proportion, and ingestion is slower (every unit is
  embedded). Timings on this loaded machine were too unstable to quote.
- **Decision at the time:** keep structural 2400 as the default (same hit rate, best MRR, simplest, fastest to ingest);
  `SemanticChunker` stays available as an option. Revisit with a larger corpus and more questions.
  (Superseded: the reverse is true on main now. `SemanticChunker` is the only chunker and the default of `ingest()`;
  the structural chunker lives only in the `chunking/structural-*` branches.)
- The grid was run with an ad-hoc script that is not part of the repository (the keyword-leg variants were
  hard-coded there); `uv run python -m rag.evaluation.report` reproduces the main table.

## Full course (310 files, 5 courses) and the three chunking techniques

The three techniques were indexed over the whole course (310 files, 309 with chunks) and evaluated on 130 questions (the 40 + 10 + 40 from course 1
with their source paths prefixed, plus `golden_courses_2_5.json`: 40 questions on courses 2 to 5), at equal returned
text (about 9,000 characters per query). Hit / MRR. The chunk counts below (755 / 1173 / 1443) come from this older run;
the E29 run further down counts 741 / 1144 / 1408 for the same three e5-small techniques, and the cause of the
difference was not verified.

| technique | chunks | over 512 tokens | first indexing | vector | hybrid (df 0.15) | hybrid (df 0.05) |
|---|---|---|---|---|---|---|
| structural 2400 | 755 | 260 (34%) | 131 s | 0.662 / 0.489 | 0.708 / 0.542 | 0.731 / 0.565 |
| semantic | 1173 | 1 | 916 s | 0.700 / 0.465 | 0.731 / 0.496 | 0.738 / 0.521 |
| structure + semantic | 1443 | 0 | 115 s (cached windows; about 15 min alone) | 0.723 / 0.500 | 0.754 / 0.493 | **0.762** / 0.562 |
| plain recursive (no headers) | 648 | 286 (44%) | 108 s | 0.677 / 0.492 | 0.685 / 0.535 | 0.700 / 0.543 |

By set (hybrid, df 0.05): course 1 tuning sets A / T / V and courses 2 to 5 (F): structural 0.600 / 0.900 / 0.700 / 0.850,
semantic 0.700 / 0.900 / 0.625 / 0.850, structure + semantic 0.725 / 1.000 / 0.650 / 0.850.

- **With five times more content the ordering is consistent but small**: structure + semantic >= semantic >= structural
  in hit rate under all three search variants, by 2 to 6 points. The standard error with 130 questions is about
  3.8 points, so this is a trend, not a proof. MRR does not follow it (structural 0.565, structure + semantic 0.562).
- **Scores on the course 1 questions dropped** (for example set A, structural: 0.70 on course 1 alone, 0.60 on the
  full index) because the other courses repeat topics and add look-alike passages. Courses 2 to 5 score 0.85 to 0.90.
- **A lower document-frequency cap (0.05) beat 0.15** for all three techniques (pooled +0.8 to +2.3 points). The code
  was 0.15 at the time (tuned on 162 chunks; now 0.05 on main).
- **The plain recursive splitter (no headers, one of the four techniques) is last in hit rate** (0.700 at df 0.05) and
  has the most chunks over the embedder limit (44%); `k` is 5 for it (larger average chunk), 6 to 10 for the others.
  The recursive chunker is not in the repository now and has no branch.
- Three of the techniques live in their own branches (`chunking/structural-e5-small`, `chunking/semantic-e5-small`,
  `chunking/structure-semantic-e5-small`) with their code, tests and a `CHUNKING.md`. The recursive baseline has no branch
  (`chunking/recursive-e5-small`, tip `66b0c8f`, was deleted).

## How structure and meaning share the cuts (structure + semantic)

Why each chunk ended, counted on a sample of 24 course 1 files (every third file, 101 chunks, median 153 tokens;
small files weigh more in the sample than in the full course):

| cause | chunks | share of the 77 real decisions |
|---|---|---|
| header (structure) | 36 | 47% |
| meaning shift (semantic) | 33 | 43% |
| token cap (440) | 8 | 10% |
| end of document (not a decision) | 24 | |

A header was reached with the current chunk still under 100 tokens 33 times, so the section was merged into the next
one instead of becoming a tiny chunk: the minimum size does as much work as the cut rules.

## Where the chunks are cut

Checked on course 1 by looking, in the source markdown, at what follows each chunk end.

| chunker | chunks | cut mid-word | cut inside a sentence |
|---|---|---|---|
| structural 2400 | 162 | 0 | 7 (4%) |
| structural 1800 | 198 | 0 | 15 (8%) |
| semantic p90, min 100 | 248 | 0 | 10 (4%) |

- No chunker splits words: the recursive splitter only falls back to characters when a single word exceeds the
  chunk size.
- The structural splitter tried paragraph, line and sentence ends (`. `, `? `, `! `, `; `) before a plain
  space, and kept a section's header attached to its first piece instead of leaving it alone in a chunk (it now lives
  only in the `chunking/structural-*` branches). On course 1
  this did not change the number of cuts or the retrieval scores (162 chunks, same hit/MRR within 0.015).
- The remaining cuts come from the PDF conversion, which sometimes inserts a paragraph break (or a page number)
  in the middle of a sentence: 22 such breaks in 2 of the 70 files. Both structural and semantic chunking inherit them.

## All techniques on one method, two embedding models (E29)

Hybrid search (df cap 0.05), 130 questions, about 9,000 characters returned per query (`k` depends on the chunk size). Hits = questions answered out of 130. Full table, settings and caveats: [experiments.md](experiments.md) E29; raw output: [runs/e29_unified_eval.log](runs/e29_unified_eval.log).

| technique | e5-small hits / MRR | bge-m3 hits / MRR |
|---|---|---|
| structural 2400 | 95 / .561 | 107 / .612 |
| semantic | 96 / .517 | 113 / .576 |
| semantic + header cuts | 99 / .557 | **114** / .580 |
| community two-stage (headers, then semantic on big sections) | 97 / .508 | 113 / .584 |
| user's hybrid (two-stage + merge of neighbours) | 95 / .552 | 109 / .592 |

- The community recipe has more hits than the user's hybrid in both models (by 2 and 4 questions); the hybrid has the slightly higher MRR. Differences under about 8 points (about 10 of the 130 questions) are not conclusive, so neither is better on this evidence.
- Among the chunkers, semantic + header cuts has the highest count in both models (e5 99 against 95-97; bge-m3 114 against 107-113), but every gap is 1 to 7 questions, inside the noise: no chunker is conclusively better. It is the default because it is the highest measured and also leads in the vector and translation rows. It is not the user's hybrid.
- bge-m3 beats e5-small on every technique (12 to 17 more questions, 9 to 13 points): conclusive. At least half of the gain comes from the 22 cross-lingual questions.
- The user's hybrid makes the largest chunks (average 1268 characters), so at equal returned text it gets the smallest `k` (7 against 10).

## Retrieval techniques on the three best chunking indexes (E30)

bge-m3, 130 questions, about 9,000 characters returned per query. Cells are questions answered out of 130 / MRR. Settings, caveats and raw output: [experiments.md](experiments.md) E30, [runs/e30_retrieval_modes.log](runs/e30_retrieval_modes.log).

| retrieval | semantic + headers (`SEMH`) | semantic (`SEM`) | community two-stage (`STSEM`) |
|---|---|---|---|
| vector | 112 / 0.631 | 107 / 0.581 | 107 / 0.595 |
| vector + ES/EN 1:1 | 116 / 0.639 | 104 / 0.548 | 105 / 0.594 |
| vector + ES/EN 2:1 | 116 / 0.638 | 105 / 0.564 | 106 / 0.588 |
| keyword | 75 / 0.355 | 77 / 0.374 | 80 / 0.351 |
| keyword + ES/EN 1:1 | 75 / 0.300 | 79 / 0.313 | 73 / 0.319 |
| keyword + ES/EN 2:1 | 73 / 0.296 | 75 / 0.312 | 76 / 0.320 |
| hybrid | 114 / 0.580 | 113 / 0.576 | 113 / 0.584 |
| hybrid + ES/EN 1:1 | 122 / 0.591 | 110 / 0.551 | 114 / 0.576 |
| hybrid + ES/EN 2:1 | 121 / 0.607 | 111 / 0.591 | 112 / 0.605 |

- Highest count in 6 of the 9 rows (every vector and hybrid row): semantic + header cuts (`SEMH`); in two of the three keyword rows it comes last (two-stage is last in the 1:1 row).
- Most questions answered: hybrid + ES/EN 1:1 on `SEMH` (122), the best measured configuration. Best MRR: vector + ES/EN on `SEMH` (.639), with no keyword leg.
- With the ES/EN translation, `SEMH` answers 11-12 more questions than the other two chunkers on vector (116 against 104/105) and 12 more than semantic-only on hybrid (122 against 110): conclusive at the 8-point rule. Against two-stage on hybrid (122 against 114) the gap is 8 questions (6 points): not conclusive. Hybrid + ES/EN against vector + ES/EN on `SEMH` is 6 questions: not conclusive. 1:1 against 2:1 is tied (1 question).
- Translating the question helps only `SEMH`; keyword alone is the weakest; RRF only fuses lists, it is not a search technique by itself.

## Returning 30 chunks (E31)

Same ten indexes and nine retrieval techniques; each query returns up to 30 chunks (about 26,000 to 47,000 characters, depending on the chunk size) and the candidate pool is 30 or 100 per search. Settings, the diagnosis of the unanswered questions and caveats: [experiments.md](experiments.md) E31; raw output: [runs/e31_k30_candidates100.log](runs/e31_k30_candidates100.log) and [runs/e31_k30_candidates30.log](runs/e31_k30_candidates30.log).

Best retrieval technique per bge-m3 index (30 chunks, 100 candidates; answered / 130 and MRR):

| chunking technique | best retrieval technique | answered / 130 | MRR | characters in 30 chunks |
|---|---|---|---|---|
| semantic + header cuts | hybrid + ES/EN 1:1 | 130 | 0.553 | about 26,356 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 2:1 | 129 | 0.589 | about 38,052 |
| structural | hybrid + ES/EN 2:1 | 129 | 0.579 | about 47,005 |
| community two-stage | hybrid + ES/EN 2:1 | 125 | 0.591 | about 27,321 |
| semantic only | hybrid + ES/EN 2:1 | 125 | 0.574 | about 31,725 |

- 130 of 130 only for semantic + header cuts with hybrid + ES/EN 1:1; the rest answer 125 to 129, inside the noise of each other.
- The unanswered questions at 10 chunks were ranking misses: all 130 evidence snippets sit complete in a chunk of the best index.
- A pool of 100 candidates helps the recall at 30 and hurts the top 10 and the MRR. The 130 questions also chose these settings, so 130 of 130 is optimistic.

## New questions (E32)

50 questions about files that no earlier question used, written by a language model and checked by script against the corpus. Same indexes and techniques as E31, 30 chunks returned. Settings, caveats and raw output: [experiments.md](experiments.md) E32; [runs/e32_heldout_candidates30.log](runs/e32_heldout_candidates30.log), [runs/e32_heldout_candidates100.log](runs/e32_heldout_candidates100.log).

Hybrid + ES/EN 1:1 on the bge-m3 indexes (answered):

| chunking technique | answered of the 130 (30 / 100 candidates) | answered of the 50 new questions at 10, 20, 30 chunks (30 candidates) | at 30 chunks (100 candidates) |
|---|---|---|---|
| semantic + header cuts | 128 / 130 | 41, 46, 48 | 46 |
| structural | 126 / 127 | 45, 48, 50 | 49 |
| user's hybrid (two-stage + merge) | 126 / 127 | 43, 47, 48 | 47 |
| semantic only | 124 / 124 | 39, 45, 46 | 43 |
| community two-stage | 123 / 125 | 42, 48, 48 | 47 |

- The 130 of 130 of E31 does not repeat: the best combination of that run answers 48 or 46 of 50 here.
- No chunking technique is conclusively better (differences under about 6 of 50 questions); a pool of 100 candidates is not better than 30; e5-small catches up with bge-m3 when the translation is fused.

## Limits

- Small sets (a single question moves hit@5 by 2.5 points on the main set; ignore differences under about 7 points).
- One correct passage per question: another file that also answers it (for example a summary) counts as a
  miss, so scores are conservative.
- LLM-written questions; paraphrasing was requested, which favours vectors over the keyword leg.
- The first sections (up to the tuning grid) use only the course 1 index (70 files); the full course has 310 files. int8 vs fp32 embeddings and larger models were not compared.
