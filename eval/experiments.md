# Experiment log

Complete record of every experiment, measurement and diagnostic run on this project, in the order they were run
(session of 2026-11-06 to 2026-11-07), including the ones that were wrong, superseded or retracted.
`results.md` is the curated summary; this file is the full ledger. Companion Spanish study guide:
`docs/GUIA_RAG_COMPLETA.pdf` (workspace root, not in this repo).

## How to read this log

- **hit@k**: share of questions for which a top-k chunk contains the cited evidence. **MRR**: mean of 1/rank of the first
  correct chunk (0 beyond k). Cells are written `hit / MRR`.
- **Equal text budget**: from E08 on, `k` is chosen per chunker so each query returns about 9,000 characters, because
  with a fixed `k` larger chunks return more text and win by coverage.
- **Noise**: one question is 2.5 points on a 40-question set and 0.77 on 130. The standard error of a hit rate near 0.75
  with 130 questions is about 4 points; differences under about 8 points (two standard errors) are not conclusive.
- **Status**: `accepted` (still holds), `superseded` (replaced by a later experiment), `retracted` (wrong, kept for the
  record), `pending`.
- **Scripts**: most experiment drivers were ad hoc scripts run from a scratch directory and are NOT in the repository.
  What is reproducible from the repo is listed in section "Reproducibility map".

## Setup common to all experiments

| Item | Value |
|---|---|
| Machine | Intel i7-1360P (16 threads), 15.7 GB RAM, no GPU, Windows 11. Budget: 20-30% of CPU and RAM |
| Corpus | Google Cloud Data Analytics certificate. Course 1: 70 markdown files (40 PDF converted + 30 native). Full: 311 files, 5 courses, about 1.8 MB of text (161 PDFs, 500 MB on disk, mostly images) |
| Database | Podman container `rag-db` (`pgvector/pgvector:pg17`, `127.0.0.1:5433`, 1 GB / 2 CPUs), PostgreSQL + pgvector 0.8.7. Throwaway databases per experiment |
| Embeddings | `multilingual-e5-small` int8 ONNX (384 d, 512 tokens, mean pooling, `query:`/`passage:` prefixes), 4 threads. From E25 also `bge-m3` int8 ONNX (1024 d, CLS pooling, no prefixes, 1024 tokens used) |
| Store schema | `halfvec(N)`, HNSW `halfvec_cosine_ops` (m=16, ef_construction=64, ef_search=100), `tsvector` ('simple') + GIN |
| Retrieval | Vector leg + keyword leg (Postgres full-text, `ts_rank_cd`, words of 3+ chars joined with OR, document-frequency cap), 30 candidates each, fused with RRF (k=60) |

Golden sets (every evidence snippet verified verbatim against the corpus files):

| Set | n | Content | Origin |
|---|---|---|---|
| A (`golden_set.json`) | 40 | Paraphrased, course 1, 22 ES / 18 EN, 14 cross-lingual | Sonnet subagent, read-only, 0 verification problems |
| T (`golden_terms.json`) | 10 | Short exact-term queries (HIPAA, IAM, Looker...) | Written by hand from corpus greps |
| V (`golden_validation.json`) | 40 | Paraphrased, course 1, 20 ES / 20 EN, not used for tuning | Sonnet subagent; 28 files, 11 also in A, 0 duplicate questions |
| F (`golden_courses_2_5.json`) | 40 | Paraphrased, courses 2-5 (12/12/12/4), 40 distinct files | Sonnet subagent |
| Translations (`golden_translations.json`) | 130 | Each question translated to the other language | Sonnet subagent (stand-in for the calling LLM) |

Test suite growth: 21 (after ingest) -> 34 (retrieval) -> 43 -> 52 -> 64 -> 67 -> 69 -> 74 -> 83 tests, all passing.
Security: Bandit 0 issues and pip-audit no known vulnerabilities, run before each important push.

---

## Phase 1: parsing, chunking baseline, ingestion

### E01. PDF to Markdown: which pymupdf4llm engine?
- **Question**: which engine gives usable text from the course PDFs.
- **Setup**: `pymupdf4llm` layout engine (default) vs legacy engine (`use_layout(False)`) vs raw `pymupdf.get_text`, on two course 1 PDFs.
- **Result**: layout engine turned ligatures into U+FFFD (13 in one 4,226-character Spanish PDF: "Certi�cado", "con�anza"). Raw extraction and
  legacy: 0. Legacy ran at 1.2-1.5 s per PDF and is lighter. Course 1 converted: 70 files in 34 s, 228 KB, 0 U+FFFD.
- **Decision**: legacy engine. **Status**: accepted. **Commit**: `b9edd96`. Later, full course: 241 new files in 77 s, 0 U+FFFD.

### E02. Structure of the converted course 1
- 70 files; files median 314 words (max 2792); 295 sections split on headers: **median 53 words, p90 222, max 1317**;
  3 files without headers, 57 with 3 or more.
- **Conclusion**: cutting only at headers would give tiny chunks, so adjacent sections are merged up to the size limit.
- **Status**: accepted. **Commit**: `cb7adda`.

### E03. Chunk counts by size (structural chunker, course 1)
- 1200 chars -> 268 chunks; 2400 -> 162; 4000 -> 104. Distribution after merge: median 1130 / 1612 / 2280 characters.
- **Status**: accepted (descriptive).

### E04. First ingestion with the real model; model file choice
- **Model file**: `Xenova/multilingual-e5-small` `model_quantized.onnx` (118 MB int8). The `intfloat/...qint8_avx512_vnni` build was rejected because
  the CPU has no AVX512-VNNI.
- **Run** (course 1, structural 2400): 70 docs -> 162 chunks in **13.8 s**; CPU avg **22.4%**, peak 24.6% of 16 threads; process RAM peak
  **1135 MB**; Postgres container 47 MB; re-run with nothing changed: 70 skipped in 0.01 s.
- **Status**: accepted (budget met). **Commits**: `6373e84`, `8c40b66`.

### E04b. Container runtime
- Docker daemon was off; `podman compose` fails (the docker-compose provider cannot reach the podman pipe); `podman run` of
  `docker.io/pgvector/pgvector:pg17` works, pgvector 0.8.7, psycopg connection verified. **Decision**: Podman, documented `podman run`.

---

## Phase 2: retrieval and its first evaluation

### E05. Retrieval smoke test (hybrid v1) on the real index
- 5 mixed ES/EN queries: **21-126 ms** per query after a 3.7 s model load. Four of five looked good.
  "¿Qué es BigQuery y para qué sirve?" ranked `014_INTRODUCTION_BIGQUERY` third: stopwords ("para", "sirve") weigh as much as "BigQuery"
  in the keyword leg (no IDF; `ts_rank_cd` is not BM25).
- **Status**: accepted (diagnosis). **Commit**: `0a087c0`; wording fix `24d02f4` (README said BM25).

### E06. First baseline matrix: 3 chunk sizes x 3 modes (set A only, keyword leg without any filter)
| size | chunks | vector | keyword | hybrid |
|---|---|---|---|---|
| 1200 | 268 | 0.700 / 0.481 | 0.350 / 0.240 | 0.450 / 0.360 |
| 2400 | 162 | 0.650 / 0.528 | 0.350 / 0.227 | 0.525 / 0.446 |
| 4000 | 104 | 0.725 / 0.559 | 0.375 / 0.199 | 0.550 / 0.408 |
- **Finding**: the first hybrid was **worse than vectors alone** at every size. **Status**: accepted as a finding; fixed by E07.

### E07. Fixing the keyword leg: stoplist, document-frequency cap, RRF weights (set A only)
Chunk statistics: 1200 -> 268 chunks, tokens median 272 / max 508, 0 over 512; 2400 -> 162, 390 / 983, 50 over 512 (31%); 4000 -> 104, 576 / 1521, 56 over 512 (54%).

| variant (hit / MRR) | 1200 | 2400 | 4000 |
|---|---|---|---|
| vector only | 0.700 / 0.481 | 0.650 / 0.528 | 0.725 / 0.559 |
| hybrid base 1:1 | 0.450 / 0.360 | 0.525 / 0.446 | 0.550 / 0.408 |
| + static ES/EN stoplist | 0.450 / 0.360 | 0.575 / 0.407 | 0.575 / 0.442 |
| df cap 0.30 | 0.550 / 0.388 | 0.600 / 0.461 | 0.650 / 0.531 |
| df cap 0.15 | 0.550 / 0.383 | **0.700 / 0.540** | 0.650 / 0.554 |
| stoplist + df 0.30 | 0.600 / 0.410 | 0.675 / 0.487 | 0.700 / 0.533 |
| base, RRF weights 2:1 | 0.500 / 0.369 | 0.575 / 0.460 | 0.600 / 0.427 |
| stoplist, 2:1 | 0.500 / 0.374 | 0.600 / 0.435 | 0.600 / 0.510 |
| df 0.15, 2:1 | 0.575 / 0.399 | 0.700 / 0.544 | 0.650 / 0.583 |
| stoplist + df 0.30, 2:1 | 0.650 / 0.453 | 0.700 / 0.523 | 0.700 / 0.575 |
| keyword only + stoplist | 0.375 / 0.261 | 0.375 / 0.244 | 0.375 / 0.231 |
| keyword only, df 0.15 | 0.375 / 0.290 | 0.525 / 0.385 | 0.500 / 0.349 |
- **Conclusion**: dropping query words found in more than 15% of chunks (and at least 5 chunks) fixed the hybrid at 2400 (0.525 -> 0.700);
  a static stoplist is inconsistent; 2:1 weights add little. **Status**: accepted, implemented in `store.keyword_search` (`3c4bb27`).
  (The 0.15 value was later found to be too high for large indexes: E21.)

### E08. Equal returned text, chunk size (set A, about 12,000 characters per query)
| size | k | chunks | over 512 | vector | hybrid (df 0.15) | avg chars returned |
|---|---|---|---|---|---|---|
| 1200 | 10 | 268 | 0 | 0.750 / 0.487 | 0.675 / 0.399 | 10350 / 10945 |
| 1800 | 7 | 198 | 25 | 0.700 / 0.528 | 0.700 / 0.517 | 9311 / 10229 |
| 2400 | 5 | 162 | 50 | 0.650 / 0.528 | 0.700 / 0.540 | 7838 / 9253 |
| 4000 | 3 | 104 | 56 | 0.600 / 0.529 | 0.600 / 0.542 | 8034 / 9229 |
- **Finding**: 4000 was clearly the worst at equal text; 1200-2400 within noise. The earlier "4000 is best" (E06) was coverage, not quality.
  **Status**: accepted. Default kept at 2400.

### E09. Modes on sets A and T with the final implementation (k=5)
| size | set | vector | keyword | hybrid |
|---|---|---|---|---|
| 1200 | A | 0.700 / 0.481 | 0.375 / 0.290 | 0.550 / 0.383 |
| 1200 | T | 0.900 / 0.592 | 1.000 / 0.900 | 1.000 / 0.733 |
| 1800 | A | 0.650 / 0.521 | 0.500 / 0.383 | 0.700 / 0.517 |
| 1800 | T | 0.900 / 0.600 | 1.000 / 0.950 | 1.000 / 0.700 |
| 2400 | A | 0.650 / 0.528 | 0.525 / 0.385 | 0.700 / 0.540 |
| 2400 | T | 0.900 / 0.628 | 1.000 / 0.950 | 1.000 / 0.833 |
- **Finding**: on exact terms the keyword leg wins (10/10); vectors miss "Google Gemini"; hybrid is robust on both. **Status**: accepted
  (the "hybrid beats vector on paraphrases" half of this was retracted by E17). Reproducible: `report` CLI. **Commits**: `bc8b097`, `5cbf826`.

### E10. Candidate ceiling (does a reranker have room?)
Hybrid, set A, `hit@k` for k = 5 / 10 / 20 / 30:
- size 1800: 0.700 / 0.700 / 0.775 / 0.800; never in the top 30: q01, q03, q06, q08, q14, q17, q24, q28.
- size 2400: 0.700 / 0.700 / 0.750 / 0.775; never in the top 30: the same plus q26.
- **Conclusion**: a perfect reranker could add at most 8-10 points, and misses are candidates that never enter the pool. **Status**: accepted for course 1;
  **not repeated on the full index**.

### E11. Why the misses? (reachability and file-level diagnosis)
- Evidence is wholly inside some chunk for every question at 1200, 1800 and 2400 (no unreachable questions).
- For 8 of 9 misses at 2400 the expected **file** is not in the top 30 at all; the first result is often a thematically close file.
  The top-1 chunk for q01, q03, q06, q08, q14, q28 was unrelated; q24 returned the summary of the same topic (a valid alternative the golden set does not credit).
- **Status**: accepted.

### E12. Language-pair breakdown (set A, 2400, hybrid / vector, hit@5)
| question -> evidence | n | vector | hybrid |
|---|---|---|---|
| EN -> EN | 15 | 14/15 | 15/15 |
| ES -> ES | 11 | 10/11 | 11/11 |
| EN -> ES | 3 | 1/3 | 1/3 |
| ES -> EN | 11 | 1/11 | 1/11 |
- **Finding**: within a language 26 of 26; across languages almost nothing. The "30% failures" are essentially cross-lingual. **Status**: accepted.

### E13. Querying in both languages (set A, 14 cross-lingual questions translated by hand, 2400)
| strategy | all 40 | cross-lingual 14 |
|---|---|---|
| original only | 0.700 / 0.540 | 0.143 / 0.054 |
| translated only (oracle: knows the document language) | 1.000 / 0.796 | 1.000 / 0.786 |
| original + translated, fused with RRF | 0.925 / 0.680 | 0.786 / 0.455 |
- Implemented as `HybridRetriever.search_many`. **Status**: accepted (translations by hand are a proxy for what the calling LLM would send). **Commit**: `3c4bb27`.

---

## Phase 3: semantic chunking

### E14. Boundary quality audits (what gets cut)
Three audits, only the last is valid.

| audit | structural 2400 | structural 1800 | semantic | structure+semantic | status |
|---|---|---|---|---|---|
| 1: line without final punctuation = mid-sentence cut | 70 (43%) | 94 (47%) | 114 (44%) | 122 (39%) | **retracted** (counts titles and bullets as cuts) |
| 2: sentence continues in the next chunk | 76 (47%) | 84 (42%) | 69 (28%) | - | **retracted** (bug: `'' in ",;"` is true, so end-of-file chunks were counted) |
| 3: same, with the bug fixed | **7 of 162 (4%)** | **15 of 198 (8%)** | **10 of 248 (4%)** | - | accepted |
- Mid-word cuts: **0** in all chunkers (first audit).
- Cause of the residual cuts: the PDF conversion inserts a paragraph break or a page number inside a sentence (22 such breaks in 2 of 70 files);
  all chunkers inherit them. Page numbers also appear glued at paragraph ends ("recorrido! 2").
- **Change made**: oversized sections are split at sentence ends before spaces, and a section header stays attached to its first piece
  (a header alone in a one-line chunk was a real defect found by a test). **Effect on course 1: none measurable** (E15b). **Commit**: `f912c1c`.

### E15. Semantic chunker, round 1 (course 1, k=5, sets A and T)
| config | chunks | tokens med/max | over 512 | avg chars | ingest | A vector | A hybrid | T hybrid |
|---|---|---|---|---|---|---|---|---|
| structural 1200 | 268 | 272 / 508 | 0 | 989 | 102 s | 0.700 / 0.481 | 0.550 / 0.383 | 1.0 / 0.733 |
| structural 2400 | 162 | 390 / 983 | 50 | 1544 | 98 s | 0.650 / 0.528 | 0.700 / 0.540 | 1.0 / 0.833 |
| semantic p90, min 100 | 257 | 249 / 493 | 0 | 994 | 267 s | 0.700 / 0.548 | 0.625 / 0.414 | 1.0 / 0.750 |
| semantic p75, min 100 | 338 | 186 / 484 | 0 | 786 | 242 s | 0.650 / 0.506 | 0.575 / 0.396 | 1.0 / 0.633 |
| semantic p90, min 200 | 217 | 298 / 493 | 0 | 1152 | 187 s | 0.650 / 0.508 | 0.650 / 0.455 | 1.0 / 0.700 |
(Ingest times measured on a loaded machine; the structural default took 14 s when idle.)
- **Finding**: at equal chunk size semantic is more precise (vs structural 1200: hybrid 0.625 vs 0.550); no semantic config beat structural 2400 in hybrid.
  **Status**: accepted. **Commits**: `3bd7110`, `8fcbbdf`, docs `0846a24`.

### E15b. Effect of the sentence-aware splitter (course 1, k=6, df 0.15, set sizes A / T / V)
- Before (E17 row S2400): vector A+T 0.740 / 0.555, V 0.725 / 0.578; hybrid A+T 0.760 / 0.598, V 0.750 / 0.498.
- After: vector A 0.700 / 0.538, T 0.900 / 0.633, V 0.725 / 0.580 (A+T 0.740 / 0.557); hybrid A 0.700 / 0.540, T 1.000 / 0.900, V 0.750 / 0.494 (A+T 0.760 / 0.612).
  Same chunk count (162); cuts inside a sentence still 7; header-only chunks 0. **Status**: accepted (correct in principle, no change here).

### E16. Semantic vs structural at equal returned text (course 1, set A / T)
| chunker | k | A vector | A hybrid | T vector | T hybrid | avg chars |
|---|---|---|---|---|---|---|
| structural 2400 | 5 | 0.650 / 0.528 | 0.700 / 0.540 | 0.900 / 0.628 | 1.0 / 0.833 | 7838-9253 |
| semantic p90 m100 | 5 | 0.700 / 0.548 | 0.625 / 0.414 | 0.900 / 0.558 | 1.0 / 0.750 | 5317-6554 |
| semantic p90 m100 | 8 | 0.750 / 0.555 | 0.675 / 0.422 | 1.0 / 0.575 | 1.0 / 0.750 | 8319-10491 |
| semantic p90 m100 | 10 | 0.750 / 0.555 | 0.700 / 0.425 | 1.0 / 0.575 | 1.0 / 0.750 | 10093-12961 |
- **Finding**: semantic + vectors only is best on paraphrases; the keyword leg tuned on 2400-char chunks adds noise to small chunks. **Status**: superseded by E17.

### E17. Grid with a validation set: 6 chunkers x 7 search variants (course 1, equal text, tune on A+T, validate on V)
Cells `hit / MRR`. Variants: vector; hybrid with df cap 0.10 / 0.15 / 0.30 and RRF weights 1:1 / 2:1.

| chunker (k, chunks) | variant | A | T | V | A+T |
|---|---|---|---|---|---|
| S2400 (6, 162) | vector | .700/.536 | .900/.628 | .725/.578 | .740/.555 |
| | hyb .10 1:1 | .650/.529 | 1.0/.833 | .775/.566 | .720/.590 |
| | hyb .10 2:1 | .675/.547 | 1.0/.817 | .725/.575 | .740/.601 |
| | hyb .15 1:1 | .700/.540 | 1.0/.833 | .750/.498 | .760/.598 |
| | hyb .15 2:1 | .700/.544 | 1.0/.817 | .725/.537 | .760/.598 |
| | hyb .30 1:1 | .600/.461 | 1.0/.833 | .675/.432 | .680/.536 |
| | hyb .30 2:1 | .625/.485 | 1.0/.817 | .650/.464 | .700/.552 |
| S1800 (7, 198) | vector | .700/.528 | .900/.600 | .700/.580 | .740/.542 |
| | hyb .10 1:1 | .675/.481 | 1.0/.700 | .725/.540 | .740/.525 |
| | hyb .10 2:1 | .650/.504 | 1.0/.750 | .725/.592 | .720/.553 |
| | hyb .15 1:1 | .700/.517 | 1.0/.700 | .700/.516 | .760/.553 |
| | hyb .15 2:1 | .700/.502 | 1.0/.750 | .700/.541 | .760/.552 |
| | hyb .30 1:1 | .625/.446 | 1.0/.700 | .650/.473 | .700/.497 |
| | hyb .30 2:1 | .625/.430 | 1.0/.750 | .625/.490 | .700/.494 |
| SEM p90 m100 (9, 257) | vector | .750/.555 | 1.0/.575 | .725/.558 | .800/.559 |
| | hyb .10 1:1 | .650/.472 | 1.0/.750 | .700/.423 | .720/.527 |
| | hyb .10 2:1 | .675/.509 | 1.0/.717 | .675/.471 | .740/.551 |
| | hyb .15 1:1 | .675/.422 | 1.0/.750 | .775/.504 | .740/.488 |
| | hyb .15 2:1 | .700/.445 | 1.0/.717 | .750/.527 | .760/.499 |
| | hyb .30 1:1 | .650/.432 | 1.0/.750 | .675/.453 | .720/.496 |
| | hyb .30 2:1 | .650/.436 | 1.0/.717 | .675/.476 | .720/.492 |
| SEM+H p90 m100 (11, 310) | vector | .725/.528 | 1.0/.512 | .750/.535 | .780/.525 |
| | hyb .10 1:1 | .700/.463 | 1.0/.733 | .775/.479 | .760/.517 |
| | hyb .10 2:1 | .700/.501 | 1.0/.667 | .775/.534 | .760/.534 |
| | hyb .15 1:1 | .700/.435 | 1.0/.733 | .775/.505 | .760/.495 |
| | hyb .15 2:1 | .700/.478 | 1.0/.667 | .775/.538 | .760/.516 |
| | hyb .30 1:1 | .625/.394 | .900/.633 | .750/.456 | .680/.442 |
| | hyb .30 2:1 | .650/.454 | .900/.567 | .750/.463 | .700/.477 |
| SEM+H p90 m200 (8, 231) | vector | .700/.528 | 1.0/.535 | .750/.526 | .760/.529 |
| | hyb .10 1:1 | .675/.487 | 1.0/.633 | .700/.527 | .740/.517 |
| | hyb .10 2:1 | .675/.498 | 1.0/.653 | .725/.549 | .740/.529 |
| | hyb .15 1:1 | .675/.448 | 1.0/.633 | .700/.521 | .740/.485 |
| | hyb .15 2:1 | .675/.461 | 1.0/.653 | .750/.582 | .740/.500 |
| | hyb .30 1:1 | .650/.424 | 1.0/.633 | .700/.504 | .720/.466 |
| | hyb .30 2:1 | .650/.448 | 1.0/.653 | .700/.527 | .720/.489 |
| SEM+H p75 m200 (9, 252) | vector | .750/.522 | 1.0/.534 | .750/.486 | .800/.525 |
| | hyb .10 1:1 | .700/.468 | 1.0/.733 | .725/.508 | .760/.521 |
| | hyb .10 2:1 | .700/.497 | 1.0/.733 | .725/.501 | .760/.544 |
| | hyb .15 1:1 | .700/.426 | 1.0/.733 | .700/.467 | .760/.487 |
| | hyb .15 2:1 | .700/.445 | 1.0/.733 | .700/.500 | .760/.503 |
| | hyb .30 1:1 | .675/.408 | 1.0/.733 | .675/.456 | .740/.473 |
| | hyb .30 2:1 | .675/.436 | 1.0/.733 | .675/.490 | .740/.495 |
- **Findings**: no chunker clearly better (hits 0.70-0.80, gaps of a few questions); the best tuning result did not carry over (semantic vector A+T 0.800 -> V 0.725);
  structural 2400 has the best MRR; df cap 0.30 is worst everywhere; weights 2:1 give no consistent gain.
- **Retraction**: on the validation set at k=5 the current hybrid (0.675 / 0.486) did **not** beat vector only (0.700 / 0.574): the hybrid advantage on paraphrases seen in E09 was partly tuning
  noise; its real value is exact terms. **Status**: accepted. **Commits**: `f5afaa8`, `033f9af`.

---

## Phase 4: the full course

### E18. Full-course conversion and indexing (three chunkers)
- Conversion: 241 new files in 77 s (course 1 skipped); 310 md files, sizes per course 393 / 442 / 306 / 542 / 85 KB; 0 U+FFFD; 26 files under 200 characters (normal short intros).
- Indexing (311 documents each):

| chunker | chunks | avg chars | tokens med/max | over 512 | first indexing |
|---|---|---|---|---|---|
| structural 2400 | 755 | 1574 | 397 / 951 | 260 (34%) | 131 s |
| semantic p90 m100 | 1173 | 1044 | 268 / 515 | 1 | 916 s |
| structure + semantic | 1443 | 882 | 217 / 501 | 0 | 115 s (reused cached window embeddings; about 15 min alone) |
| recursive only (E20) | 648 | 1745 | 493 / 935 | 286 (44%) | 108 s |

### E19. Full-index evaluation of the three chunkers (130 questions, equal text, hit / MRR)
| chunker (k) | variant | A | T | V | F | pooled |
|---|---|---|---|---|---|---|
| structural (6) | vector | .600/.488 | .600/.350 | .625/.436 | .775/.577 | .662/.489 |
| | hybrid df .15 | .625/.489 | .900/.592 | .550/.387 | .900/.738 | .708/.542 |
| | hybrid df .05 | .600/.440 | .900/.592 | .700/.512 | .850/.738 | .731/.565 |
| semantic (9) | vector | .625/.458 | .800/.306 | .600/.414 | .850/.564 | .700/.465 |
| | hybrid df .15 | .625/.429 | .900/.500 | .650/.435 | .875/.623 | .731/.496 |
| | hybrid df .05 | .700/.445 | .900/.500 | .625/.457 | .850/.667 | .738/.521 |
| structure+semantic (10) | vector | .700/.510 | .900/.327 | .600/.416 | .825/.618 | .723/.500 |
| | hybrid df .15 | .700/.414 | 1.0/.520 | .600/.430 | .900/.629 | .754/.493 |
| | hybrid df .05 | .725/.497 | 1.0/.514 | .650/.480 | .850/.722 | .762/.562 |
- **Findings**: order in hit rate structure+semantic >= semantic >= structural under all three search variants (2-6 points; a trend within about 1 standard error);
  MRR does not follow (structural .565 vs structure+semantic .562 at df .05); hybrid beats vector in all three; scores on course 1 questions fall with
  the full index (look-alike passages from other courses); courses 2-5 score 0.85-0.90; **df cap 0.05 beats 0.15 for all three** (+0.8 to +2.3 points).
  **Status**: accepted. The code still uses 0.15 (open item). **Commits**: `bfc178c`, `4809081`.

### E20. Plain recursive baseline (never tested before; the user assumed it had been)
- Course 1 (138 chunks, avg 1729 chars, tokens 478 / 908, 52 over 512 = 38%, ingest 22 s), k=5, pooled over A+T+V: vector 0.711 / 0.537, hybrid df .15 0.700 / 0.552, df .05 0.678 / 0.537
  (by set at df .15: A .650/.476, T 1.0/.900, V .675/.540).
- Full index (648 chunks, 286 over 512 = 44%, ingest 108 s), k=5:

| variant | A | T | V | F | pooled |
|---|---|---|---|---|---|
| vector | .625/.428 | .600/.270 | .675/.500 | .750/.604 | .677/.492 |
| hybrid df .15 | .600/.454 | .900/.717 | .550/.444 | .850/.662 | .685/.535 |
| hybrid df .05 | .600/.443 | .900/.717 | .625/.466 | .825/.677 | .700/.543 |
- **Finding**: last in hit rate on the full index (0.700 vs 0.731 / 0.738 / 0.762), suggesting that using headers helps. A header can end alone in a chunk when the next paragraph exceeds the size
  limit (documented by a test; 0 of 648 chunks under 60 characters on the real course); no header path in the chunk. **Status**: accepted.
  **Commits**: `46bca68`, `a9e0e3b`; branch `chunking/recursive-e5-small` (`8495c16`).

### E21. How structure and meaning share the cuts (structure + semantic, sample of 24 course 1 files, 101 chunks, median 153 tokens)
| cause | chunks | share of 77 real decisions |
|---|---|---|
| header (structure) | 36 | 47% |
| meaning shift (semantic) | 33 | 43% |
| token cap (440) | 8 | 10% |
| end of document (not a decision) | 24 | - |
- A header was reached with the current chunk under 100 tokens 33 times (merged into the next section instead of making a tiny chunk).
- **Status**: accepted (small sample, orientative). **Commit**: `b97c5e5`.

### E22. Is the 512-token truncation hiding answers? (structural 2400, full index)
- 128 of 130 questions have their evidence inside the part of the chunk the model reads; **2 (2%) only in the truncated tail**, and both were found.
  Vector, visible: 84/128 = 0.656; hybrid df .05, visible: 93/128 = 0.727; tail-only: 2/2 for both.
- **Conclusion**: truncation does not explain the small differences between chunkers (the keyword leg also sees the whole chunk). **Status**: accepted.

---

## Phase 5: embedding models and retrieval variants

### E23. Same model for index and query? (e5 index, 130 questions, k=6)
| query vectors | result |
|---|---|
| 1024-d vector against the 384-d index | database error: different halfvec dimensions 384 and 1024 |
| from the same space as the index | hit@6 0.662 / MRR 0.489 |
| same dimension, different space (random orthogonal rotation: **simulation**, not a real second model) | hit@6 **0.000** / 0.000 (guessing: 0.008) |
- **Conclusion**: stored vectors and the query vector must come from the same model. The model that decides chunk cuts can differ from the retrieval model.
  **Status**: accepted. **Open**: the database does not record which model produced its vectors.

### E24. Resource measurement, e5-small vs bge-m3 (60 real chunks, 4 threads, int8 ONNX, same script)
| | e5-small | bge-m3 |
|---|---|---|
| model file | 113 MB | 542 MB |
| dimension / tokens read | 384 / 512 | 1024 / 1024 (model supports 8192) |
| load time | 0.8 s | 3.8 s |
| RAM before -> after load -> peak | 69 -> 444 -> 912 MB | 69 -> 880 -> 1591 MB |
| 60 chunks | 10.6 s = 5.6 chunks/s = 1737 tokens/s | 96.4 s = 0.6 chunks/s = 205 tokens/s |
| CPU avg / peak (of 16 threads) | 22.1% / 27.0% | 23.0% / 27.4% |
| query latency median / p95 | 12 / 18 ms | 59 / 71 ms |
| ES question vs EN passages: related / unrelated cosine | 0.876 / 0.707 | 0.743 / 0.212 |
- **Conclusion**: bge-m3 fits the budget (peak 1.6 GB, same CPU because of the thread cap) but is about 9x slower; one phrase pair is not a retrieval result. **Status**: accepted.
  Measured RAM includes the Python process; estimated whole-course indexing time scales with tokens (e5 about 2 min, consistent with the measured 131 s).

### E25. e5-small on the full index: same-language vs cross-lingual, and ES+EN fusion (130 questions: 108 same-language, 22 cross-lingual)
| index (k) | search | total | same-language | cross-lingual |
|---|---|---|---|---|
| structural (6) | vector, 1 phrasing | .662 / .489 | .769 | .136 |
| | vector + RRF (ES+EN) | .646 / .376 | .648 | .636 |
| | hybrid, 1 phrasing | .731 / .565 | .861 | .091 |
| | hybrid + RRF (ES+EN) | .754 / .456 | .750 | .773 |
| structure+semantic (10) | vector, 1 phrasing | .723 / .500 | .824 | .227 |
| | vector + RRF (ES+EN) | .769 / .403 | .769 | .773 |
| | hybrid, 1 phrasing | .762 / .562 | .870 | .227 |
| | hybrid + RRF (ES+EN) | .800 / .458 | .787 | .864 |
- **Findings**: fusing the two languages lifts cross-lingual from 0.09-0.23 to 0.64-0.86 but adds noise to same-language questions (-5 to -21 points, MRR down); the keyword leg still adds 3-4 points.
  RRF needs two lists: "RRF + vector only" is only meaningful as RRF over several vector queries. **Status**: accepted. **Commit**: `b4ac48b` (translations).

### E26. bge-m3 on the full index, structural chunking (755 chunks, 130 questions, k=6, df .05)
- Indexing: **1134 s** (about 19 min) vs 131 s with e5; tokens median 395 / max 949; 257 over 512, none over 1024.

| search | total | A | T | V | F | same-language | cross-lingual |
|---|---|---|---|---|---|---|---|
| vector, 1 phrasing | .785 / .599 | .825/.645 | .600/.475 | .700/.503 | .875/.682 | .787 / .635 | .773 / .425 |
| vector + RRF (ES+EN) | .792 / .572 | - | - | - | - | .787 / .586 | .818 / .502 |
| hybrid, 1 phrasing | .808 / .609 | .775/.550 | .800/.650 | .750/.534 | .900/.731 | .880 / .691 | .455 / .206 |
| hybrid + RRF (ES+EN) | **.846** / .585 | - | - | - | - | .843 / .588 | .864 / .566 |
- **Findings**: bge-m3 improves every variant over e5-small on the same index (vector +12 points, hybrid +8, best +9); fixes most cross-lingual with one phrasing (0.773 vs 0.136);
  single-phrasing hybrid hurts cross-lingual (0.455) because the keyword leg only matches within a language; the keyword leg still adds 2-5 points and matters for exact terms (T: vector .600 vs hybrid .800).
  **Status**: accepted (structural chunking only). **Commit**: `29f0f82` (configurable embedder).

### E27. bge-m3 with structure + semantic chunking
- **Question**: does bge-m3 beat e5-small when both use structure + semantic chunking, and by how much?
- **Setup**: full corpus (311 files, 309 docs indexed), 130 questions (A 40, T 10, V 40, F 40), k = 10 (about 9,000 characters returned), hybrid with df cap 0.05, single phrasing.
  bge-m3 (int8 ONNX, 1024 dims) both cuts (`SemanticChunker(bge, 440, 100, 90, split_on_headers=True)`) and embeds. Database `full_SEMH_bge`: 1416 chunks, average 879 chars, tokens median 213 / max 499, none over 512.
  Scripts: `bge_run.py` and `resume_semh.py` (session scratchpad, **not in the repo**). The indexing was interrupted when the window closed (177 of 309 docs done) and resumed without dropping the database; the resumed run added 132 docs in 1194 s.
- **Result** (hit rate / MRR; e5 values from E28 and the branch `CHUNKING.md`):

| search | e5-small (1443 chunks) | bge-m3 (1416 chunks) | difference (hit) |
|---|---|---|---|
| vector, total | .723 / .500 | **.862** / .631 | +13.9 points |
| hybrid (df .05), total | .762 / .562 | **.877** / .580 | +11.5 points |
| vector, same-language (108) | .824 | .870 | +4.6 |
| vector, cross-lingual (22) | .227 | .818 | +59.1 |
| hybrid, same-language (108) | .870 | .907 | +3.7 |
| hybrid, cross-lingual (22) | .227 | .727 | +50.0 |

  bge-m3 per set, vector / hybrid: A .950/.692 / .925/.564, T .900/.404 / 1.000/.510, V .700/.482 / .775/.527, F .925/.774 / .900/.666.
- **Conclusion**: bge-m3 is clearly better on this corpus: +12 to +14 points in total hit rate, far above the 3.8-point standard error. Most of the gain is cross-lingual (Spanish question, English passage and the reverse); same-language improves 4-5 points.
  The keyword leg still helps exact terms (T: vector .900 vs hybrid 1.000) but hurts cross-lingual (.818 to .727), as in E26. On the structural index bge-m3 gave .808 hybrid (E26); structure + semantic gives .877, about 7 points more.
- **Caveats**: (1) the chunks are not identical: bge-m3 also decides the cuts, so the gain mixes better cutting with better retrieval; separating them needs cutting with one model and embedding with the other (`same_model.py`-style run, pending). (2) Only 22 cross-lingual questions: that subset is noisy (one question is 4.5 points). (3) The ES+EN fusion and the weighted fusion (E28) were not run on this index. (4) Resumed indexing: documents are deterministic, so the result should equal an uninterrupted run, but this was not checked.
- **Status**: accepted (single phrasing, vector and hybrid). **Commit**: this entry; the scripts are not in the repo.

### E28. Weighted RRF over two phrasings: more weight on the original question (130 questions)
Weights are original : translation; the translation always comes from `golden_translations.json`. Cells `hit / MRR`.

| index, search | 1 phrasing | RRF 1:1 | RRF 2:1 | RRF 3:1 |
|---|---|---|---|---|
| e5 structural (k=6), vector: total | .662/.489 | .646/.376 | .623/.383 | .631/.386 |
| same-language / cross-lingual (hit) | .769 / .136 | .648 / .636 | .704 / .227 | .713 / .227 |
| e5 structural, hybrid: total | .731/.565 | .754/.456 | .723/.454 | .738/.463 |
| same-language / cross-lingual (hit) | .861 / .091 | .750 / .773 | .815 / .273 | .833 / .273 |
| e5 structure+semantic (k=10), vector: total | .723/.500 | .769/.403 | .715/.384 | .723/.394 |
| same-language / cross-lingual (hit) | .824 / .227 | .769 / .773 | .806 / .273 | .815 / .273 |
| e5 structure+semantic, hybrid: total | .762/.562 | .800/.453 | .738/.455 | .738/.469 |
| same-language / cross-lingual (hit) | .870 / .227 | .787 / .864 | .833 / .273 | .833 / .273 |
| bge-m3 structural (k=6), vector: total | .785/.599 | .792/.572 | .808/.601 | .800/.604 |
| same-language / cross-lingual (hit) | .787 / .773 | .787 / .818 | .806 / .818 | .796 / .818 |
| bge-m3 structural, hybrid: total | .808/.609 | .846/.585 | .838/.611 | .838/.618 |
| same-language / cross-lingual (hit) | .880 / .455 | .843 / .864 | .843 / .818 | .843 / .818 |
- **Finding**: with e5-small, weighting the original recovers same-language accuracy but **destroys the cross-lingual gain** (0.864 -> 0.273 for hybrid, 0.773 -> 0.273 for vector): only 1:1 keeps it, because the
  translated query is the only one that finds those answers and a lower weight buries it. With bge-m3 the model already crosses languages, so the translation adds about 3 points in total (0.808 -> 0.838-0.846) and the weights hardly matter;
  3:1 gives the best MRR (0.618) at a total hit of 0.838. A second phrasing was essential with e5-small and is optional with bge-m3.
- **Status**: accepted (22 cross-lingual questions; translations by a model, as a stand-in for the caller). Run by an ad hoc script (not in the repo).

---

### E29. All chunking techniques on one common method (10 indexes, 130 questions, hybrid)
- **Question**: which chunking technique wins when every index is measured the same way, and does the user's hybrid (community two-stage recipe + merge step) beat the community recipe alone?
- **Setup**: full corpus (310 files, 309 with chunks), 130 questions (A 40, T 10, V 40, F 40), single phrasing, hybrid = vector + keyword leg fused with RRF, df cap 0.05.
  `k` is set per index so every query returns about 9,000 characters (`k = round(9000 / average chunk length)`), which gives larger-chunk techniques fewer results. One script (`eval_all.py`, session scratchpad, **not in the repo**) evaluated all ten databases in one run; the raw output is `eval/runs/e29_unified_eval.log`.
  Chunker settings: structural `StructuralChunker(2400, 200)` (characters); semantic `SemanticChunker(emb, 440, 100, 90, split_on_headers=False)`; semantic + header cuts the same with `split_on_headers=True`; **community two-stage** `StructureThenSemanticChunker(emb, 440, 100, 90)`; **user's hybrid** `StructureSemanticMergeChunker(emb, 440, 100, 25)`.
- **Result** (hybrid, pooled over 130 questions; hits = questions with a relevant chunk in the top k):

| chunking technique | model | database | chunks | avg chars | k | hits / 130 | hit rate | MRR | same-lang (108) | cross-lingual (22) |
|---|---|---|---|---|---|---|---|---|---|---|
| structural | e5-small | `full_S2400` | 741 | 1567 | 6 | 95 | .731 | .561 | .861 | .091 |
| semantic | e5-small | `full_SEM` | 1144 | 1045 | 9 | 96 | .738 | .517 | .852 | .182 |
| semantic + header cuts | e5-small | `full_SEMH` | 1408 | 883 | 10 | 99 | .762 | .557 | .870 | .227 |
| **community two-stage** | e5-small | `full_STSEM_e5` | 1335 | 911 | 10 | 97 | .746 | .508 | .870 | .136 |
| **user's hybrid (two-stage + merge)** | e5-small | `full_STMERGE_e5` | 907 | 1267 | 7 | 95 | .731 | .552 | .861 | .091 |
| structural | bge-m3 | `full_S2400_bge` | 741 | 1567 | 6 | 107 | .823 | .612 | .898 | .455 |
| semantic | bge-m3 | `full_SEM_bge` | 1128 | 1058 | 9 | 113 | .869 | .576 | .917 | .636 |
| semantic + header cuts | bge-m3 | `full_SEMH_bge` | 1416 | 879 | 10 | **114** | **.877** | .580 | .907 | .727 |
| **community two-stage** | bge-m3 | `full_STSEM_bge` | 1336 | 911 | 10 | 113 | .869 | .584 | .926 | .591 |
| **user's hybrid (two-stage + merge)** | bge-m3 | `full_STMERGE_bge` | 906 | 1268 | 7 | 109 | .838 | .592 | .917 | .455 |

  Vector only (hits / 130): structural e5 86, semantic e5 92, semantic + header cuts e5 94, two-stage e5 91, hybrid e5 89; structural bge 102, semantic bge 107, semantic + header cuts bge 112, two-stage bge 107, hybrid bge 105.
- **Findings**:
  - Community vs user's hybrid: the community two-stage recipe has more hits in both models (e5 97 vs 95, bge 113 vs 109), the user's hybrid has a higher MRR in both (e5 .552 vs .508, bge .592 vs .584). The gaps are 2 and 4 questions out of 130, below the ~4 point standard error: **not conclusive**.
  - The best index overall is semantic + header cuts with bge-m3 (114 hits). That is **not** the user's hybrid (see the technique definitions in `CLAUDE.md`).
  - bge-m3 beats e5-small on every technique (+12 to +17 hits); about half or more of the gain is the 22 cross-lingual questions (e5 gets 2-5 of them, bge-m3 10-14).
  - Semantic + header cuts makes more chunks than pure semantic because it adds a third cut: at every `#` header, when the current chunk already has 100 tokens (`starts_section and tokens >= min_tokens`). 1416 vs 1128 chunks with bge-m3.
  - The user's hybrid makes the biggest chunks (average 1268 characters), so at equal returned text it gets the smallest `k` (7). Whether it would close the gap at `k = 10` was not measured.
- **Caveats**: single run per index; `k` differs by technique; the chunk counts of the three e5 indexes (741, 1144, 1408) differ from the ones in the older branch documents (755, 1173, 1443) while the hit counts match, and the cause was **not verified** (the working tree has uncommitted changes in `parse.py`); `full_STMERGE_bge` was first evaluated with 289 of 309 documents and `full_SEM_bge` with 69, then both were completed (resume with the same chunker settings: 20 and 241 documents added, the rest skipped) before the numbers above.
- **Status**: accepted (single run, n = 130). **Branches**: one per technique and model, each with a `CHUNKING.md` (see "Branches and commits").

## Corrections log (things said or measured wrongly, and what replaced them)

| # | Claim or measurement | Reality | Where fixed |
|---|---|---|---|
| 1 | The keyword leg is "BM25" | It is Postgres `ts_rank_cd`, no IDF | README (`24d02f4`), E07 |
| 2 | The first hybrid improves on vectors | It was worse (E06); fixed by the df cap (E07) | `3c4bb27` |
| 3 | "Hybrid beats vector on paraphrased questions" | Not replicated on the validation set (E17); its value is exact terms | `033f9af`, results.md |
| 4 | Chunk size "2400 tokens" | 2400 **characters**; real tokens median 397, max 951; 34% over the 512-token limit | E19, E22 |
| 5 | First mid-sentence audits: 43-47% | Heuristic and a Python bug; real value 4% (E14) | E14 |
| 6 | Compare sizes at the same `k` | Larger chunks win by returning more text; redone at equal text (E08) | E08 |
| 7 | Structural + recursive vs recursive only | Recursive alone had never been run; implemented and measured (E20) | E20 |
| 8 | df cap 0.15 | Tuned on 162 chunks; 0.05 is better on 750-1450 chunks (E19). Code unchanged yet | open |
| 9 | Reranker not needed | Measured only on course 1 (E10); not repeated on the full index | open |
| 10 | "Vectors alone miss exact terms almost always" | 9 of 10 on course 1; 0.600 vs 0.900 hybrid on the full index | PDF guide |
| 11 | First unified run of E29: 0.000 on sets A, T and V for every index | Evaluation bug, not retrieval: the course 1 golden sets lack the `01_INTRODUCTION_GC_DATA_ANALYTICS/` prefix that the indexes have (`load_golden(path, source_prefix)`); redone with the prefix. `python -m rag.evaluation.report` still loads those sets without the prefix | E29 |

## Operational notes

- `rag-db` would not start: it mounted `rag/sql/init.sql`, which no longer exists. Recreated with `podman run` on the same volume `rag_pgdata` and without that mount (the script only runs on first initialisation); all ten databases were intact.
- Tooling quirk: `\n` escapes written through shell heredocs were turned into real newlines inside source files (broke `chunk.py` twice); use the editor tool for code with backslash escapes.
- The machine slowed 5-10x for periods (background apps): indexing times are noisy; quote only the ones marked as measured idle.
- `uv init` created a nested git repo inside `rag/`, which became the project repo (remote `estvn/GCP-ANALYTICS_RAG`); `mcp/` and `mcp-drawio/` have their own empty repos.
- Skills installed globally from skills.sh and audited before install: `langchain-rag`, `rag-implementation`, `pgvector-semantic-search`, `mcp-builder`,
  `python-project-structure`, `auditing-python-security`, `python-performance-optimization`, `postgres`, `python-clean-code`.
  Not installed: NVIDIA RAG skills (GPU/NIM stack), generic vector-database skills, `secure-coding`, `scalability-playbook`.

## Reproducibility map

| Reproducible from this repository | Ad hoc (script not in the repo) |
|---|---|
| `uv run python -m rag.evaluation.report [--model e5-small\|bge-m3]` (vector / keyword / hybrid on the golden sets) | The chunker x size grids (E06-E08, E15-E17), the full-index runs (E18-E22, E25-E26), the resource and rotation tests (E23-E24) |
| `ingest(md_root, conn, embedder, chunker=...)` with `RecursiveChunker`, `StructuralChunker`, `SemanticChunker`; `HybridRetriever.search` / `search_many` | Boundary audits (E14), tail-truncation test (E22), cut-cause counting (E21) |
| Golden sets and translations under `eval/` (verified evidence) | |
| 83 automated tests (`uv run pytest`) | |

## Branches and commits

Branches: `main` (all techniques), `chunking/recursive-e5-small` (`8495c16`), `chunking/structural-e5-small` (`4d4352b`), `chunking/semantic-e5-small` (`c418c67`), `chunking/structure-semantic-e5-small` (`84c5341`).
Main commits in order: `c499977` scaffold, `b9edd96` parse, `cb7adda` chunk, `6373e84` embed, `8c40b66` ingest, `4f5fb05` refactor, `6de9c0a` README, `0a087c0` retrieval,
`24d02f4` docs, `8655b7e` golden set, `3c4bb27` keyword df cap + search_many, `bc8b097` report + terms set, `5cbf826` results, `8fcbbdf` chunker injection, `3bd7110` semantic chunker,
`0846a24` docs, `f5afaa8` structure-first option, `033f9af` validation + grid docs, `f912c1c` sentence-aware splitter, `bfc178c` courses 2-5 set,
`4809081` full-course docs, `46bca68` recursive chunker, `a9e0e3b` docs, `b97c5e5` cut-cause docs, `29f0f82` configurable embedder, `b4ac48b` translations.

## Open experiments and decisions

1. ES+EN fusion and weighted fusion (E28) on the bge-m3 structure + semantic index (E27 is done), and a run that cuts with one model and embeds with the other.
2. Decide the final configuration (current leader on the data so far: bge-m3 + structural or structure + semantic + hybrid with ES+EN RRF) and keep only that path on `main`; techniques stay in their branches.
3. Branches `embedding/e5-small` and `embedding/bge-m3` and a results document per technique.
4. Lower `MAX_DOC_FREQUENCY` (0.15 -> 0.05); store the embedding model name in the database.
5. Re-measure the reranker ceiling on the full index; fixed-size and clustering chunkers were never tested; int8 vs full precision never compared.
6. MCP server (`mcp/`) and draw.io MCP (`mcp-drawio/`).
