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
| Corpus | Google Cloud Data Analytics certificate. Course 1: 70 markdown files (40 PDF converted + 30 native). Full: 310 files (309 with chunks), 5 courses, about 1.8 MB of text (161 PDFs, 500 MB on disk, mostly images) |
| Database | Podman container `rag-db` (`pgvector/pgvector:pg17`, `127.0.0.1:5433`, 1 GB / 2 CPUs), PostgreSQL + pgvector 0.8.7. Throwaway databases per experiment |
| Embeddings | `multilingual-e5-small` int8 ONNX (384 d, 512 tokens, mean pooling, `query:`/`passage:` prefixes), 4 threads. From E26 also `bge-m3` int8 ONNX (1024 d, CLS pooling, no prefixes, 1024 tokens used) |
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

Test suite growth: 21 (after ingest) -> 34 (retrieval) -> 43 -> 52 -> 64 -> 67 -> 69 -> 74 -> 83 tests, all passing. Today the suite has 73 tests (83 was the count while the structural/recursive chunker tests existed).
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
- Indexing (310 documents each; the chunk counts below are from this older run, E29 counts 741 / 1144 / 1408 for the same techniques, cause not verified):

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
  **Status**: accepted. The code used 0.15 at the time (done since: `MAX_DOC_FREQUENCY` is 0.05 on main). **Commits**: `bfc178c`, `4809081`.

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
  **Commits**: `46bca68`, `a9e0e3b`; branch `chunking/recursive-e5-small` (`8495c16`; deleted since, the recursive chunker is no longer in the repository).

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
- **Setup**: full corpus (310 files, 309 docs indexed), 130 questions (A 40, T 10, V 40, F 40), k = 10 (about 9,000 characters returned), hybrid with df cap 0.05, single phrasing.
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
- **Caveats**: (1) the chunks are not identical: bge-m3 also decides the cuts, so the gain mixes better cutting with better retrieval; separating them needs cutting with one model and embedding with the other (`same_model.py`-style run, pending). (2) Only 22 cross-lingual questions: that subset is noisy (one question is 4.5 points). (3) The ES+EN fusion and the weighted fusion (E28) were not run on this index (the ES/EN fusion was later done in E30; the weighted fusion also ran there). (4) Resumed indexing: documents are deterministic, so the result should equal an uninterrupted run, but this was not checked.
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
- **Question**: which chunking technique answers the most questions when every index is measured the same way, and does the user's hybrid (community two-stage recipe + merge step) beat the community recipe alone?
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
  - Community vs user's hybrid: the community two-stage recipe has more hits in both models (e5 97 vs 95, bge 113 vs 109), the user's hybrid has a higher MRR in both (e5 .552 vs .508, bge .592 vs .584). The gaps are 2 and 4 questions out of 130; differences under about 8 points (about 10 of the 130 questions) are not conclusive, so this is **not conclusive**.
  - Among the chunkers with the plain hybrid, semantic + header cuts has the highest count in both models (e5 99 against 95-97; bge-m3 114 against 107-113), but every gap is 1 to 7 questions, inside the noise: **no chunker is conclusively better**. It is the default of `main` because it is the highest measured and also leads in the vector and translation rows (E30). It is **not** the user's hybrid (see the technique definitions in `CLAUDE.md`).
  - bge-m3 beats e5-small on every technique (+12 to +17 hits, 9 to 13 points): **conclusive**. About half or more of the gain is the 22 cross-lingual questions (e5 gets 2-5 of them, bge-m3 10-14).
  - Semantic + header cuts makes more chunks than pure semantic because it adds a third cut: at every `#` header, when the current chunk already has 100 tokens (`starts_section and tokens >= min_tokens`). 1416 vs 1128 chunks with bge-m3.
  - The user's hybrid makes the biggest chunks (average 1268 characters), so at equal returned text it gets the smallest `k` (7). Whether it would close the gap at `k = 10` was not measured.
- **Caveats**: single run per index; `k` differs by technique; the chunk counts of the three e5 indexes (741, 1144, 1408) differ from the ones in the older branch documents and in E19 (755, 1173, 1443) while the hit counts match, and the cause was **not verified** (the working tree has uncommitted changes in `parse.py`); `full_STMERGE_bge` was first evaluated with 289 of 309 documents and `full_SEM_bge` with 69, then both were completed (resume with the same chunker settings: 20 and 241 documents added, the rest skipped) before the numbers above.
- **Status**: accepted (single run, n = 130). **Branches**: one per technique and model, each with a `CHUNKING.md` (see "Branches and commits").

### E30. Retrieval techniques on the three best chunking indexes (bge-m3, 130 questions)
- **Question**: how do vector, keyword and hybrid search compare on the three most precise chunking techniques, and does fusing the question with its translation (ES/EN RRF) help each of them?
- **Setup**: indexes `full_SEMH_bge` (semantic + header cuts), `full_SEM_bge` (semantic), `full_STSEM_bge` (community two-stage); embedding model bge-m3; the 130 questions; `k` from the 9,000-character budget (10, 9, 10). Techniques: vector (cosine over halfvec, HNSW), keyword (Postgres full-text `ts_rank_cd`, document-frequency cap 0.05), hybrid (vector + keyword, 30 candidates each, RRF k = 60). ES/EN RRF runs the same search with the question and its hand translation (`eval/golden_translations.json`) and fuses the two lists with weights `original:translation` of 1:1 or 2:1; score = weight / (60 + position). Scripts `retrieval_modes.py` and `retrieval_modes2.py` (session scratchpad, **not in the repo**); raw output `eval/runs/e30_retrieval_modes.log`.
- **Result** (questions answered out of 130 / MRR):

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

- **Findings**:
  - `SEMH` has the highest count in 6 of the 9 rows (every vector and hybrid row) and is last in two of the three keyword rows (rows 1 and 3; two-stage is last in the 1:1 row). The gaps are conclusive in some rows and not in others (next bullet). The hybrid with ES/EN 1:1 on `SEMH` answers the most questions (122), the best measured configuration; vector with ES/EN on `SEMH` has the best MRR (0.639) with 116 answers and no keyword leg.
  - Translation helps only `SEMH` (vector 112 to 116, hybrid 114 to 122); on `SEM` and `STSEM` it is neutral or worse.
  - Keyword alone is the weakest (75 to 80 answered). Translation does not raise its total: it rescues cross-lingual questions (`SEMH`: .091 to .818 with 1:1) and loses same-language ones (.676 to .528).
  - RRF is not a search technique by itself: it only fuses lists; here it was applied on top of vector, keyword and hybrid.
  - With the ES/EN translation, `SEMH` answers 11-12 more questions than the other two chunkers on vector (116 against 104/105) and 12 more than semantic-only on hybrid (122 against 110): conclusive at the 8-point rule. Against two-stage on hybrid (122 against 114) it is 8 questions (6 points): not conclusive.
  - The 6-question gap between hybrid + ES/EN and vector + ES/EN on `SEMH` is not conclusive (under about 8 points). Vector leads in MRR. 1:1 against 2:1 is tied (1 question).
- **Caveats**: single run; the second phrasing is a hand translation, so a real system would need a translation step (extra call, two searches per query), which this does not measure; the keyword leg uses a document-frequency cap of 0.05 and the scripts of this run are scratch files (the 2:1 weights and the translations option are now implemented in the branches `retrieval/hybrid-es-en-rrf` and `retrieval/vector-es-en-rrf`), while the repo default of `MAX_DOC_FREQUENCY` is 0.05 (changed after this run; it was 0.15) and `search_many` uses unweighted `rrf` (1:1); the user's hybrid chunking and the e5-small indexes were not run through these modes.
- **Status**: accepted (single run, n = 130). **Branches**: `retrieval/vector`, `retrieval/keyword`, `retrieval/hybrid`, and the two ES/EN RRF variants that did best, `retrieval/hybrid-es-en-rrf` and `retrieval/vector-es-en-rrf`.

### E31. Returning 30 chunks and a larger candidate pool (10 indexes, 130 questions)
- **Question**: how many of the 130 questions are answered when a query returns 30 chunks instead of about 10, and what does the size of the candidate pool (chunks taken from each search before fusion) change?
- **Setup**: the ten indexes and nine retrieval techniques of E29-E30. Every query returns up to 30 chunks (`k = 30`); two pool sizes, 30 candidates per search (the default of `HybridRetriever`) and 100. Each phrasing is searched on its own, its list is cut at the pool size, and the lists are fused (hybrid: vector + keyword; ES/EN: the lists of the two phrasings). A question is answered when a returned chunk contains its evidence (the E29 rule). `@10`, `@20` and `@30` count the questions answered within the first 10, 20 and 30 chunks of the same ranking. Scripts `matrix_k30.py` and `diagnose_misses.py` (session scratchpad, **not in the repo**); raw output `eval/runs/e31_k30_candidates30.log` and `eval/runs/e31_k30_candidates100.log`.
- **Result** (best retrieval technique per index, 30 chunks, 100 candidates; answered / 130 and MRR of the first 30 chunks):

bge-m3:

| chunking technique | best retrieval technique | answered / 130 | MRR | characters in 30 chunks |
|---|---|---|---|---|
| semantic + header cuts | hybrid + ES/EN 1:1 | 130 | 0.553 | about 26,356 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 2:1 | 129 | 0.589 | about 38,052 |
| structural | hybrid + ES/EN 2:1 | 129 | 0.579 | about 47,005 |
| community two-stage | hybrid + ES/EN 2:1 | 125 | 0.591 | about 27,321 |
| semantic only | hybrid + ES/EN 2:1 | 125 | 0.574 | about 31,725 |

e5-small:

| chunking technique | best retrieval technique | answered / 130 | MRR | characters in 30 chunks |
|---|---|---|---|---|
| user's hybrid (two-stage + merge) | hybrid + ES/EN 1:1 | 121 | 0.387 | about 38,009 |
| structural | hybrid + ES/EN 1:1 | 121 | 0.380 | about 47,005 |
| semantic only | hybrid + ES/EN 1:1 | 120 | 0.367 | about 31,363 |
| community two-stage | hybrid + ES/EN 1:1 | 120 | 0.367 | about 27,336 |
| semantic + header cuts | hybrid + ES/EN 1:1 | 119 | 0.386 | about 26,489 |

  All nine techniques on the best index (`full_SEMH_bge`, semantic + header cuts):

| retrieval technique | @10 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|
| vector | 112 | 125 | 112 | 122 | 125 | 0.637 |
| vector + ES/EN 1:1 | 116 | 126 | 117 | 123 | 127 | 0.645 |
| vector + ES/EN 2:1 | 116 | 125 | 115 | 124 | 127 | 0.644 |
| keyword | 75 | 94 | 75 | 88 | 94 | 0.365 |
| keyword + ES/EN 1:1 | 75 | 102 | 73 | 89 | 102 | 0.297 |
| keyword + ES/EN 2:1 | 73 | 96 | 74 | 91 | 98 | 0.312 |
| hybrid | 114 | 127 | 105 | 121 | 126 | 0.580 |
| hybrid + ES/EN 1:1 | 122 | 128 | 119 | 127 | 130 | 0.553 |
| hybrid + ES/EN 2:1 | 121 | 127 | 122 | 127 | 129 | 0.601 |

- **Where the unanswered questions go** (diagnosis on `full_SEMH_bge`, hybrid + ES/EN 1:1, 100 candidates per search): the evidence of all 130 questions sits complete inside at least one chunk (0 unreachable), so no chunker cut through an evidence snippet. Questions answered within the first 5, 10, 20 and 30 chunks: 97, 119, 127 and 130. The 11 questions missed at 10 chunks had their first correct chunk between positions 11 and 30: they are a ranking problem, not a chunking problem. This diagnosis was run on that index only.
- **At 20 chunks** (the same rankings cut at position 20; about 17,600 characters for semantic + header cuts): the best retrieval technique answers 127 of 130 on `full_SEMH_bge` and on `full_S2400_bge` (31,300 characters), 126 on the user's hybrid, 125 on semantic only and 123 on two-stage (bge-m3); the best e5-small index answers 117 to 118 with 30 candidates. The 3 questions that `full_SEMH_bge` still misses at 20 chunks have their first correct chunk at positions 22, 24 and 30: they are beyond 20 by construction, since 30 chunks answer all 130.
- **Findings**:
  - Returning 30 chunks instead of about 10 answers more questions in every combination. On `full_SEMH_bge` with hybrid + ES/EN 1:1: 122 answered at 10 chunks and 128 at 30 with 30 candidates; 130 at 30 with 100 candidates.
  - A larger pool helps the recall at 30 but hurts the top of the ranking: hybrid on `full_SEMH_bge` goes from 114 to 105 answered at 10 chunks, and hybrid + ES/EN 1:1 loses MRR (0.594 to 0.553) because deeper keyword lists add noise to the fusion.
  - Only one combination answers all 130: semantic + header cuts, hybrid + ES/EN 1:1, 30 chunks, 100 candidates. The next ones answer 129 (structural and the user's hybrid, both hybrid + ES/EN 2:1) and 125 (semantic only and two-stage). Gaps of 1 to 5 questions are inside the noise: at 30 chunks no chunking technique is conclusively better. What differs is the text returned: about 26,000 characters for semantic + header cuts, 38,000 for the user's hybrid and 47,000 for structural.
  - bge-m3 stays ahead of e5-small technique by technique (5 to 11 questions): the best e5-small index answers 121 of 130.
  - Keyword search alone stays last (94 to 100 answered at 30 chunks with bge-m3 indexes).
  - Vector search has the best MRR in the best index (0.645 with ES/EN 1:1 against 0.553 for the hybrid).
- **Caveats**: single run; the second phrasing is a hand translation; the 130 questions were also used to choose chunker, `k` and pool size, so 130 of 130 is optimistic for unseen questions and is not a promise of answering every question in use; 30 chunks of about 880 characters are about 26,000 characters, three times the text of the earlier runs, which a language model must read; no reranker was tried (E10 measured its room only on course 1); the diagnosis of unreachable evidence covers one index; differences under about 8 points are not conclusive.
- **Status**: accepted (single run, n = 130). **Branches**: every technique branch has the section "Results returning 30 chunks (E31)".

### E32. Questions nobody used to choose a setting (50 new questions, 10 indexes)
- **Question**: do the choices made on the 130 questions (chunker, `k`, pool size, ES/EN fusion) hold on new questions about the course content?
- **Setup**: 54 course files that no earlier question used as evidence (picked per course with a fixed seed: 10, 14, 12, 14 and 4 files), one question each, written by two language-model drafts and then verified by a script: 4 questions were dropped because their file is a list of citations, 50 remain (33 in Spanish, 17 in English, 12 asked in the other language than the file). Each question has its evidence copied verbatim from the file (checked by script against the corpus; 10 questions share the same evidence text with a second file, and any of the files counts) and a translation written by the draft for the ES/EN fusion. Same ten indexes, nine retrieval techniques, 30 chunks returned and pools of 30 and 100 candidates as in E31. Files `eval/golden_heldout.json` and `eval/golden_heldout_translations.json`; scripts `make_assignments.py`, `build_heldout.py` and `heldout_eval.py` (session scratchpad, **not in the repo**); raw output `eval/runs/e32_heldout_candidates30.log` and `eval/runs/e32_heldout_candidates100.log`.
- **Result** (hybrid + ES/EN 1:1, answered; bge-m3 first, then e5-small):

| chunking technique | answered of the 130 (30 / 100 candidates) | answered of the 50 new questions at 10, 20, 30 chunks (30 candidates) | at 30 chunks (100 candidates) |
|---|---|---|---|
| semantic + header cuts | 128 / 130 | 41, 46, 48 | 46 |
| structural | 126 / 127 | 45, 48, 50 | 49 |
| user's hybrid (two-stage + merge) | 126 / 127 | 43, 47, 48 | 47 |
| semantic only | 124 / 124 | 39, 45, 46 | 43 |
| community two-stage | 123 / 125 | 42, 48, 48 | 47 |

| chunking technique | answered of the 130 (30 / 100 candidates) | answered of the 50 new questions at 10, 20, 30 chunks (30 candidates) | at 30 chunks (100 candidates) |
|---|---|---|---|
| structural | 122 / 121 | 45, 49, 50 | 50 |
| user's hybrid (two-stage + merge) | 122 / 121 | 46, 49, 50 | 49 |
| semantic only | 121 / 120 | 42, 45, 47 | 45 |
| community two-stage | 118 / 120 | 46, 47, 47 | 48 |
| semantic + header cuts | 117 / 119 | 45, 47, 48 | 47 |

  All nine techniques on `full_SEMH_bge` (semantic + header cuts), 50 new questions:

| retrieval technique | @10 (30 cand.) | @20 (30 cand.) | @30 (30 cand.) | @10 (100 cand.) | @20 (100 cand.) | @30 (100 cand.) | MRR (100 cand.) |
|---|---|---|---|---|---|---|---|
| vector | 39 | 42 | 44 | 39 | 42 | 44 | 0.534 |
| vector + ES/EN 1:1 | 37 | 44 | 44 | 37 | 43 | 44 | 0.505 |
| vector + ES/EN 2:1 | 37 | 44 | 44 | 37 | 44 | 44 | 0.527 |
| keyword | 32 | 36 | 38 | 32 | 36 | 38 | 0.471 |
| keyword + ES/EN 1:1 | 41 | 42 | 44 | 36 | 42 | 44 | 0.444 |
| keyword + ES/EN 2:1 | 34 | 37 | 39 | 33 | 36 | 39 | 0.460 |
| hybrid | 39 | 46 | 46 | 38 | 41 | 46 | 0.516 |
| hybrid + ES/EN 1:1 | 41 | 46 | 48 | 42 | 44 | 46 | 0.583 |
| hybrid + ES/EN 2:1 | 42 | 46 | 46 | 41 | 45 | 46 | 0.544 |

- **Findings**:
  - The 130 of 130 was optimistic. On the 50 new questions the same combination (semantic + header cuts, hybrid + ES/EN 1:1, 30 chunks) answers 48 with 30 candidates and 46 with 100, against 128 and 130 of 130 on the questions that chose the settings.
  - No chunking technique is conclusively better. The best technique of every index answers 45 to 50 of 50 with 100 candidates; with 50 questions one question is 2 points and differences under about 6 questions are not conclusive. The structural bge-m3 index answers 50 of 50 with 30 candidates, semantic + header cuts 48. The earlier lead of semantic + header cuts is not confirmed; what it keeps is the smaller text (about 26,000 characters in 30 chunks, against 47,000 for structural).
  - Over all 180 questions the top combinations are within two questions of each other: structural with hybrid + ES/EN 2:1, 100 candidates: 178/180; user's hybrid (two-stage + merge) with hybrid + ES/EN 2:1, 100 candidates: 177/180; structural with hybrid + ES/EN 1:1, 30 candidates: 176/180; semantic + header cuts with hybrid + ES/EN 1:1, 30 candidates: 176/180.
  - A pool of 100 candidates is not better than 30 on the new questions: with hybrid + ES/EN 1:1, 30 candidates answer the same or more in 9 of 10 indexes (strictly more in 8); over the 180 questions the best indexes tie (176 and 176). It also lowers the top 10 and the MRR (E31).
  - e5-small catches up with bge-m3 once the translation is fused: its best index answers up to 50 of 50, and it answers 11 or 12 of the 12 questions asked in the other language. The 5 to 11 question lead of bge-m3 on the 130 questions is not seen here, and 50 questions cannot separate them.
  - Keyword search alone stays last (36 to 41 of 50 at 30 chunks).
- **Caveats**: 50 questions and a single run; the questions and their translations were written by a language model, not by a person, so they may be easier or phrased differently from real questions; only 12 questions cross languages; the evidence snippets are short; the files differ from those of the 130 questions, so topics differ too. The defaults of `search` and `search_many` (30 chunks, 100 candidates) were set after E31; this experiment finds no gain from the pool of 100.
- **Status**: accepted (single run, n = 50). **Branches**: every technique branch has the section "Results on 50 new questions (E32)".

### E33. Which embedding model, read from E31 and E32 (analysis of existing runs)
- **Question**: how do e5-small and bge-m3 compare when more chunks are returned, and which one fits production?
- **Setup**: no new run. The raw output of E31 (130 questions) and E32 (50 new questions), 30 candidates per search, 10, 20 and 30 chunks counted from the same ranking, five chunking techniques per model, averaged over the five techniques and over the 180 questions (MRR weighted by the number of questions). Computed by a script of the session scratchpad (not in the repository) from `eval/runs/e31_k30_candidates30.log` and `eval/runs/e32_heldout_candidates30.log`.
- **Result** (share of the 180 questions answered):

| retrieval | model | first 10 chunks | first 20 | first 30 | MRR |
|---|---|---|---|---|---|
| hybrid + ES/EN 1:1 (with translation) | e5-small | 83.4% | 91.7% | 93.6% | 0.49 |
| hybrid + ES/EN 1:1 (with translation) | bge-m3 | 87.9% | 94.8% | 96.3% | 0.58 |
| hybrid (one phrasing, no translation) | e5-small | 76.9% | 81.9% | 85.4% | 0.55 |
| hybrid (one phrasing, no translation) | bge-m3 | 86.7% | 93.8% | 95.1% | 0.58 |

- **Findings**:
  - Going from 10 to 30 chunks gains 8 to 10 points for both models.
  - With the translation fused, e5-small answers 91.7% with 20 chunks, more than bge-m3 with 10 (87.9%): it needs about twice the chunks (and the text). Without the translation it does not reach bge-m3's 10 chunks even with 30.
  - The chunks are practically the same with both models (E29 counts: 741 and 741, 1,408 and 1,416, 1,335 and 1,336, 907 and 906), and the gap remains on the structural index, which uses no model to cut: the difference comes from the embeddings, not from the chunking (E24 also measured a related/unrelated similarity gap of 0.17 for e5-small and 0.53 for bge-m3).
  - bge-m3 needs no translation step; e5-small does.
- **Caveats**: the gaps with the translation (2.7 to 4.5 points) are inside the noise of 180 questions (about 6 points, about 10 questions); no end-to-end test of the answer of a language model; text measured in characters; the questions of E32 were written by a language model.
- **Decision (the user)**: the ES/EN translation fusion is always used, with both models; the rows without translation stay in this entry only as the record of the analysis.
- **Status**: accepted (analysis of single runs). **Recommendation**: bge-m3 for production; e5-small when RAM or indexing time are the limit.

### E34. Resource cost of one retrieval with each model, and the scale it implies
- **Question**: how many resources does a retrieval with bge-m3 use compared with e5-small, and what changes with many more chunks?
- **Setup**: `bench_retrieval.py` (session scratchpad, **not in the repo**) on the indexes `full_STMERGE_bge` (906 chunks) and `full_STMERGE_e5` (907 chunks); the 50 questions of E32 and their translations (100 queries); 4 CPU threads; a warm-up of 6 embeddings; the time of each step of a query and of `search_many` with its defaults (30 chunks, 100 candidates); process memory with `psutil`; table and index sizes with `pg_total_relation_size` and `pg_relation_size`. The machine was not idle, so the times are higher than those of E24 (12 and 59 ms for embedding one query), but the ratio between the models is the same (4.7 against 4.9).
- **Result**:

| measured now, real questions against the real index (906 and 907 chunks) | e5-small | bge-m3 | ratio |
|---|---|---|---|
| vector length | 384 numbers | 1,024 numbers | 2.7 times |
| size of one vector in the table | 776 bytes | 2,052 bytes | 2.6 times |
| HNSW index per chunk | 1,183 bytes | 2,740 bytes | 2.3 times |
| whole table + indexes per chunk (text, vector, HNSW, full-text) | 7,135 bytes | 9,973 bytes | 1.4 times |
| process memory after loading the model (peak) | 699 MB (715 MB) | 1,135 MB (1,152 MB) | 1.6 times |
| time to load the model | 4.4 s | 5.6 s | 1.3 times |
| embed one query, median (p95) | 32 ms (45 ms) | 150 ms (230 ms) | 4.7 times |
| vector search, 100 candidates, median | 15 ms | 20 ms | 1.3 times |
| keyword search, 100 candidates, median | 28 ms | 32 ms | 1.2 times |
| `search_many` with 2 phrasings, median (p95) | 136 ms (182 ms) | 424 ms (557 ms) | 3.1 times |

  Estimate for more chunks (the bytes per chunk and the indexing speed of E24 multiplied; an extrapolation, not a measurement):

| chunks | whole table + indexes, bge-m3 / e5-small | vectors + HNSW index only, bge-m3 / e5-small | indexing time, bge-m3 / e5-small |
|---|---|---|---|
| 10,000 | 0.1 GB / 0.1 GB | 0.05 GB / 0.02 GB | 4.6 h / 0.5 h |
| 100,000 | 1.0 GB / 0.7 GB | 0.48 GB / 0.20 GB | 46.3 h / 5.0 h |
| 1,000,000 | 10.0 GB / 7.1 GB | 4.79 GB / 1.96 GB | 463.0 h / 49.6 h |

- **Findings**: the vectors are 2.7 times longer (not 5 times); one chunk takes 1.4 times more space because the text and the full-text index dominate; the time of a query is dominated by embedding it with the model (4.7 times slower), not by PostgreSQL; model time and memory do not depend on the number of chunks; storage and indexing time grow linearly with it; with bge-m3 the vectors and their index take 2.4 times more space and indexing takes 9 times longer.
- **Caveats**: about 900 chunks, so the PostgreSQL search times at hundreds of thousands of chunks were not measured; the sizes per chunk are those of a small index and may not scale exactly linearly; indexing speed comes from 60 chunks; whether the quality gap between the models grows with the corpus was not tested.
- **Status**: accepted (single run).

### E35. Where the memory of a bge-m3 process goes
- **Question**: does loading bge-m3 always use 1.1 GB, and what is in it?
- **Setup**: `bench_memory.py` and `bench_memory2.py` (session scratchpad, **not in the repo**): a clean Python process, `onnxruntime` session with the same options as `Embedder` (4 intra-op threads, 1 inter-op), process memory (`psutil`) after each step, 100 different questions (the 50 of E32 and their translations) one at a time. Each variant run twice. A first run (60 repetitions of one single query) seemed to show that `enable_mem_pattern=False` made the query 4.5 times faster; with 100 different queries it makes no difference (142 against 149 ms) and the vectors are identical (maximum difference 0.0), so that effect was an artifact of repeating one query.
- **Result**:

| step (bge-m3, one process, measured twice with the same result) | memory | added |
|---|---|---|
| Python with the libraries imported | 51 MB | |
| ONNX session loaded (the 568 MB int8 model) | 614 MB | +563 MB |
| first tokenizer (the one that prepares the queries) | 862 MB | +248 MB |
| second tokenizer (the one that only counts tokens for the chunker) | 1,117 MB | +254 MB |
| after 100 different queries | 1,126 MB | +9 MB |

- **Findings**: the memory does not grow with queries (+9 MB over 100 queries) and does not depend on the number of chunks; about 563 MB are the model and about 500 MB the two copies of the 250,000-token tokenizer; the second copy is only used by `count_tokens`, which the chunker calls and a query does not. The 699 MB of e5-small in E34 are mostly the same two tokenizers (113 MB of model), so the memory difference between the models is the weights, about 450 MB. Loading the counting tokenizer on demand would save about 254 MB in a query process (not implemented).
- **Caveats**: one machine; memory is the working set of the process (`rss`); the figure for indexing (E24: peak 1,591 MB) was measured with another script; the saving of the lazy tokenizer is a measurement of the tokenizer, not of the changed code.
- **Status**: accepted (single run, repeated twice).

### E36. e5-small against bge-m3 on the semantic + header cuts index, by number of chunks
- **Question**: which model answers more questions with fewer chunks on the chunker that `main` uses?
- **Setup**: indexes `full_SEMH` (e5-small) and `full_SEMH_bge` (bge-m3); hybrid + ES/EN 1:1; the 130 original questions and the 50 new ones of E32 (180 in total); the rank of the first correct chunk was kept for every question and the cut made at k = 1, 3, 5, 8, 10, 15, 20, 25, 30; pools of 30 and 100 candidates per search. Script `curve_semh.py` (session scratchpad, **not in the repo**); raw output `eval/runs/e36_curve_semh.log`; the run was repeated and gave the same numbers.
- **Result** (30 candidates; answered of 180):

| chunks returned | e5-small | bge-m3 | difference |
|---|---|---|---|
| 1 | 60 (33%) | **79** (44%) | +19 |
| 3 | 111 (62%) | **121** (67%) | +10 |
| 5 | 127 (71%) | **136** (76%) | +9 |
| 10 | 146 (81%) | **163** (91%) | +17 |
| 15 | 161 (89%) | **167** (93%) | +6 |
| 20 | 164 (91%) | **172** (96%) | +8 |
| 30 | 165 (92%) | **176** (98%) | +11 |

| coverage | chunks e5-small needs | chunks bge-m3 needs |
|---|---|---|
| 85% | 12 | 8 |
| 90% | 17 | 10 |
| 95% | >30 | 18 |

  With 100 candidates: e5-small needs 16, 23 and >30 chunks for 85%, 90% and 95%, bge-m3 8, 11 and 19; MRR 0.438 against 0.561.
- **Findings**: on this index bge-m3 reaches 90% with 10 chunks and e5-small with 17; e5-small does not reach 95% within 30 chunks. The gap at 10 chunks (+17 questions) is conclusive; at 15 to 20 chunks it is inside the noise of 180 questions. A pool of 100 candidates hurts e5-small much more than bge-m3. The average over the five chunking techniques (E33) hides this: semantic + header cuts is among the places where e5-small is weakest (tied with two-stage).
- **Caveats**: single run; the 130 original questions also chose the settings; only the position of the correct chunk was measured; one machine.
- **Status**: accepted (single run, repeated once). **Recommendation**: bge-m3 for the chunker of `main`.

### E37. Only the questions written in Spanish
- **Question**: if the questions are always in Spanish, does anything change, and how do the two models compare?
- **Setup**: the 96 questions in Spanish among the 180 (the 130 original ones and the 50 new ones of E32); 21 of them have their answer in a passage in English (language of the passage detected by word counts). Index `full_SEMH` with e5-small and `full_SEMH_bge` with bge-m3, 30 candidates per search, hybrid with the Spanish question only and hybrid + ES/EN 1:1 (the question and its English translation). Script `spanish_only.py` (session scratchpad, **not in the repo**); raw output `eval/runs/e37_spanish_only.log`.
- **Result** (answered):

| search | model | all 96 Spanish questions: first 10 / 20 / 30 chunks | the 21 whose answer is in an English passage |
|---|---|---|---|
| hybrid, Spanish question only | e5-small | 72 / 75 / 78 | 3 / 4 / 5 |
| hybrid + ES/EN 1:1 | e5-small | 84 / 90 / 91 | 18 / 21 / 21 |
| hybrid, Spanish question only | bge-m3 | 81 / 88 / 91 | 14 / 18 / 19 |
| hybrid + ES/EN 1:1 | bge-m3 | 86 / 91 / 93 | 20 / 21 / 21 |

- **Findings**: with the translation fused, the two models answer almost the same number of Spanish questions (91 and 93 of 96 with 30 chunks, 84 and 86 with 10); without it the gap is large (78 and 91 with 30 chunks) because of the 21 questions whose answer is in English (e5-small answers 5, bge-m3 19). The translation is what makes Spanish questions find the English passages. Of the 11-question gap of E36 (176 against 165 of 180), 2 questions are among the Spanish ones and 9 among the 84 written in English.
- **Caveats**: 96 questions, a single run; differences of 2 questions are noise; the language of a passage is a word-count heuristic; about one Spanish question in five has its answer in English, which is a property of this corpus.
- **Status**: accepted (single run, n = 96).

### E38. The cost of a multilingual reranker
- **Question**: what would it cost to reorder the candidates with a local cross-encoder reranker?
- **Setup**: `bge-reranker-v2-m3` from `onnx-community/bge-reranker-v2-m3-ONNX` (the base model `BAAI/bge-reranker-v2-m3` is Apache-2.0; that copy declares no license), file `onnx/model_int8.onnx` (544 MB), in `models/bge-reranker-v2-m3/` (git-ignored); XLM-RoBERTa, 24 layers, hidden size 1,024, vocabulary of 250,002 tokens. `rerank_probe.py` (session scratchpad, **not in the repo**) on the index `full_SEMH_bge`, 4 questions of E32 with their translations, 30 candidates from `search_many`, 4 CPU threads, batches of 10 pairs; memory with `psutil`. Run twice. Raw output of the second run: `eval/runs/e38_reranker_cost.log`. A multilingual check: the correct passage scored higher than a wrong one in 4 of 4 questions with an English passage and 6 of 6 with a Spanish passage.
- **Result**:

| per query, 4 CPU threads | without reranker | with `bge-reranker-v2-m3` (int8 ONNX) |
|---|---|---|
| search (`search_many`, 2 phrasings) | 0.15 to 0.5 s | the same search, plus: |
| reorder 10 candidates | | + 11 to 17 s |
| reorder 30 candidates | | + 16 to 65 s (34 to 54 s in a first run) |
| memory of the process after loading | 1,135 MB (bge-m3 alone, E35) | 1,942 MB (bge-m3 and the reranker) |
| memory while scoring 30 pairs | | 3,607 MB |
| model files on disk | 542 MB | 542 MB + 544 MB |

- **Findings**: reordering 30 candidates takes tens of seconds on this CPU, hundreds of times the search itself; the process needs about 3 times the memory of bge-m3 alone while scoring. The variation between runs (16 to 65 s for 30 pairs) comes from a machine that was not idle and from the length of the pairs (274 to 353 tokens on average).
- **Caveats**: 4 questions; one machine, 4 threads (more threads or a GPU would be faster); quality was not measured: a test on 40 questions was started and stopped on purpose when the cost was clear; a smaller multilingual reranker (for example `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, 113 MB) was not tried.
- **Status**: accepted as a cost measurement. **Decision (the user)**: no reranker; rank fusion and 30 chunks stay.

### E39. A second course in the same index (GCP + Power BI), checked with the GCP questions
- **Question**: does adding a second, different course to an index make the answers for the first course worse?
- **Setup**: the Power BI course (`COURSES/GOOGLE_CLOUD_DATA_ANALYSIS_CERTIFICATE/POWERBI`: 169 PDFs and 4 Markdown files) was converted with `convert_corpus` into `data/md/POWERBI/`. 63 of its PDFs were byte-identical copies (the same file in `DOCUMENTACION/` and in `PDFs/`, one name with `+` and one with spaces); the copies in `PDFs/` were deleted from the source folder, leaving 110 documents. Two new indexes were built from scratch with the default chunker (semantic + header cuts): `courses_bge` (bge-m3, 1,658 chunks) and `courses_e5` (e5-small, 1,650 chunks), each with the 310 GCP documents and the 110 Power BI ones (420 in total); ingestion took 240 s with e5-small. Retrieval: hybrid + ES/EN 1:1, 30 chunks, 100 candidates. The 130 questions of E31 and the 50 of E32, run with `matrix_k30.py` and `heldout_eval.py` (session scratchpad, **not in the repo**; raw output was not saved) against the new index and, as a baseline, against the GCP-only index of the same model (`full_SEMH_bge`, `full_SEMH`). The questions are about GCP only.
- **Result** (questions answered within 10 / 20 / 30 chunks, and MRR@30):

| model | set | GCP only | GCP + Power BI |
|---|---|---|---|
| bge-m3 | 130 questions | 119 / 127 / 130, MRR 0.553 | 120 / 126 / 129, MRR 0.539 |
| bge-m3 | 50 questions | 42 / 44 / 46, MRR 0.583 | 42 / 44 / 46, MRR 0.551 |
| bge-m3 | **180 together** | 161 / 171 / 176 | 162 / 170 / 175 |
| e5-small | 130 questions | 100 / 115 / 119, MRR 0.386 | 93 / 113 / 121, MRR 0.377 |
| e5-small | 50 questions | 42 / 45 / 47, MRR 0.571 | 42 / 45 / 46, MRR 0.552 |
| e5-small | **180 together** | 142 / 160 / 166 | 135 / 158 / 167 |

- **Findings**: with 30 chunks the answers do not change beyond the noise (about 6 questions for n = 180): bge-m3 176 against 175, e5-small 166 against 167. With 10 chunks bge-m3 is the same (161 against 162) and e5-small answers 7 fewer (142 against 135), at the edge of the noise. MRR goes down a little in all four rows (0.01 to 0.03). A question about Dataproc returned no Power BI chunk among its 30; one about `CALCULATE` returned Power BI chunks only.
- **Caveats**: only GCP questions were measured; there is no golden set for the Power BI course, so how well it is answered is **not measured** (two spot checks only). Which chunks take the first places from the GCP ones was not investigated. The two courses share no overlapping topic that is hard to tell apart, so a corpus with closer topics could behave differently. The keyword search keeps its document-frequency cap (0.05) calibrated on the GCP course. Single run each.
- **Status**: accepted (single run, n = 180). Dumps: `data/backups/courses_bge.dump` (7.8 MB) and `courses_e5.dump` (3.1 MB), git-ignored.

### E40. Questions about the Power BI course (39 questions, 2 indexes)
- **Question**: is the Power BI part of the mixed index answered well, and which retrieval technique suits it?
- **Setup**: after E39, three more Power BI PDFs were found to have identical text to another one (a copy named with `(1)` or with a different accent) and were deleted from the source folder, so the indexes now hold 417 documents (107 of Power BI): `courses_bge` 1,655 chunks and `courses_e5` 1,647. 40 questions were drafted by two read-only sub-agents, each over disjoint folders of the converted course (`data/md/POWERBI`), one question per file, with the evidence copied from the file and an English translation. `build_powerbi_set.py` (session scratchpad, **not in the repo**) checked every evidence snippet: it must be in its own file and in no other, one question per file; one question was dropped because its evidence did not match the file (an HTML entity), leaving 39 (`eval/golden_powerbi.json` and `eval/golden_powerbi_translations.json`). It also found that the evidence of `pb16` is split between two chunks in `courses_e5`, so that index cannot answer it. All questions are in Spanish and the course is in Spanish (no cross-language question). Run with `powerbi_eval.py` (scratchpad, **not in the repo**; raw output not saved): 30 chunks, 100 candidates, same code as E31 and E32.
- **Result** (questions answered within 10 / 20 / 30 chunks, and MRR@30, of 39):

| technique | bge-m3 | e5-small |
|---|---|---|
| vector | 37 / 39 / 39, MRR 0.825 | 35 / 37 / 37, MRR 0.743 |
| vector + ES/EN 1:1 | 37 / 39 / 39, MRR 0.797 | 35 / 36 / 37, MRR 0.600 |
| hybrid | 34 / 37 / 38, MRR 0.720 | 35 / 36 / 37, MRR 0.680 |
| **hybrid + ES/EN 1:1** (the default) | 36 / 38 / 38, MRR 0.580 | 33 / 33 / 37, MRR 0.553 |

- **Findings**: with 30 chunks the default answers 38 of 39 with bge-m3 and 37 of 39 with e5-small, so the Power BI course is found in the first 30 chunks. The ranking inside those 30 is better with the plain vector search (MRR 0.825 against 0.580 with bge-m3): on same-language questions the translation fusion and the keyword leg push the correct chunk down. This is the same effect as in E31 (the fusion helps the questions written in the other language of the file, and costs ranking elsewhere), here without any cross-language question to compensate.
- **Caveats**: 39 questions, one run, so a difference of 1 or 2 answered questions is noise; the MRR gap is large enough to be real for this set but it was not repeated. The questions were drafted from the same text they are checked against, so their wording may echo the evidence and favour the vector search. They only cover files with enough extracted text: several Power BI PDFs are slides and gave almost no text when converted (some under 100 characters), so **what those slides say is not in the index and this set cannot see it**; OCR was not tried. The 05_IA_POWERBI folder has no file with enough text to ask about. The dumps `data/backups/courses_*.dump` were regenerated after the last deduplication.
- **Status**: accepted (single run, n = 39). Decision pending: the default (hybrid + ES/EN 1:1) stays for now, since the questions of the course can come in either language.

### E41. OCR of the Power BI slides that had no text
- **Question**: how much of the Power BI course was missing because its slides are screenshots, and does adding what OCR reads from them change the answers?
- **Setup**: the 103 Power BI PDFs have 372 pages; 187 of them have under 200 characters in their text layer (counted with `pymupdf`; every PDF has at least one such page). Those pages were rendered at 110 dpi and read with RapidOCR (`rapidocr-onnxruntime`, local ONNX, 4 threads; `ocr_all.py` in the session scratchpad, **not in the repo**; raw lines in `ocr_raw.json`, same place). The text of a page was appended to the Markdown of its file under `## Texto de las imágenes (página N)` only when the OCR had more than 150 characters and more than 1.5 times the characters of the text layer (title slides are skipped), after dropping lines with fewer than 3 letters or digits. The 44 changed files were re-ingested (`updated: 44`; 31 s with e5-small, 249 s with bge-m3). The golden sets of E39 (180 GCP questions) and E40 (39 Power BI questions) were run again, same technique and parameters.
- **Result**: 187 pages read, 75,267 characters of OCR text in all (including the title slides that were not kept); 72 pages in 44 files were appended; each index grew by 73 chunks (`courses_bge` 1,655 to 1,728; `courses_e5` 1,647 to 1,720).

| hybrid + ES/EN 1:1, answered within 10 / 20 / 30 | before OCR | with OCR |
|---|---|---|
| bge-m3, 180 GCP questions | 162 / 170 / 175 | 161 / 171 / 175 |
| e5-small, 180 GCP questions | 135 / 158 / 167 | 136 / 158 / 168 |
| bge-m3, 39 Power BI questions (MRR@30) | 36 / 38 / 38 (0.580) | 35 / 37 / 38 (0.587) |
| e5-small, 39 Power BI questions (MRR@30) | 33 / 33 / 37 (0.553) | 30 / 33 / 35 (0.520) |

- **Findings**: nothing changes beyond the noise, except that e5-small answers 2 Power BI questions fewer (37 to 35; 39 questions, so about noise). In a sample of 10 low-text pages (before the full run), 5 were title slides where OCR added nothing and 5 were screenshots where it read 400 to 1,000 characters: some explanatory sentences (for example, how anomaly detection words its explanation) and a lot of interface text. The OCR does not read accents and sometimes joins words.
- **Caveats**: the 39 questions were written from files that already had text, so this set cannot show a gain from text that only the screenshots have: **the benefit of the OCR is not measured**. A set written from the screenshots would be needed. Single run; 110 dpi (not compared with a higher resolution); one OCR engine (no Tesseract or other model was tried). Title-slide pages are not in the index.
- **Status**: accepted as a measurement (no gain, no clear loss). Decision pending whether to keep the OCR text in the published indexes.

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
| 8 | df cap 0.15 | Tuned on 162 chunks; 0.05 is better on 750-1450 chunks (E19) | `MAX_DOC_FREQUENCY` = 0.05 in `main` |
| 9 | Reranker not needed | Measured only on course 1 (E10); not repeated on the full index | open |
| 10 | "Vectors alone miss exact terms almost always" | 9 of 10 on course 1; 0.600 vs 0.900 hybrid on the full index | PDF guide |
| 11 | First unified run of E29: 0.000 on sets A, T and V for every index | Evaluation bug, not retrieval: the course 1 golden sets lack the `01_INTRODUCTION_GC_DATA_ANALYTICS/` prefix that the indexes have (`load_golden(path, source_prefix)`); redone with the prefix. `python -m rag.evaluation.report` still loads those sets without the prefix | E29 |

## Operational notes

- `rag-db` would not start: it mounted `rag/sql/init.sql`, which no longer exists. Recreated with `podman run` on the same volume `rag_pgdata` and without that mount (the script only runs on first initialisation); all ten databases were intact.
- The indexes `full_STMERGE_bge` and `full_STMERGE_e5` (the author's hybrid chunking with each model) are dumped to `rag/data/backups/` (git-ignored); both dumps were restored into a temporary database and had the same chunks and documents as the originals. The other indexes live only in the `rag_pgdata` volume.
- Tooling quirk: `\n` escapes written through shell heredocs were turned into real newlines inside source files (broke `chunk.py` twice); use the editor tool for code with backslash escapes.
- The machine slowed 5-10x for periods (background apps): indexing times are noisy; quote only the ones marked as measured idle.
- `uv init` created a nested git repo inside `rag/`, which became the project repo (remote `estvn/GCP-ANALYTICS_RAG`); `mcp/` and `mcp-drawio/` have their own empty repos.
- Skills installed globally from skills.sh and audited before install: `langchain-rag`, `rag-implementation`, `pgvector-semantic-search`, `mcp-builder`,
  `python-project-structure`, `auditing-python-security`, `python-performance-optimization`, `postgres`, `python-clean-code`.
  Not installed: NVIDIA RAG skills (GPU/NIM stack), generic vector-database skills, `secure-coding`, `scalability-playbook`.
- Skills added for the MCP server, each read in full and copied from a pinned commit: `uv` (`astral-sh/claude-code-plugins@f3ce88a`, the authors of uv) and `pytest`
  (`gentleman-programming/gentleman-skills@c8036a3`, Apache-2.0). No FastMCP skill was installed: the most installed one (`jezweb/claude-skills`) is no longer in its
  repository, and the other (`ovachiever/droid-tings`, last updated 2025-11) predates FastMCP 4 and ships deployment scripts; the server was built with `mcp-builder`
  and the FastMCP documentation instead. No pydantic skill: the official one covers Pydantic AI and instrumentation only.

## Reproducibility map

| Reproducible from this repository | Ad hoc (script not in the repo) |
|---|---|
| `uv run python -m rag.evaluation.report [--model e5-small\|bge-m3]` (vector / keyword / hybrid on the golden sets) | The chunker x size grids (E06-E08, E15-E17), the full-index runs (E18-E22, E25-E26), the resource and rotation tests (E23-E24) |
| `ingest(md_root, conn, embedder, chunker=...)` with `SemanticChunker` (the highest measured, the only chunker on main and the default); `HybridRetriever.search` / `search_many`. The structural, recursive, two-stage and merge chunkers are only in their branches | Boundary audits (E14), tail-truncation test (E22), cut-cause counting (E21) |
| Golden sets and translations under `eval/` (verified evidence) | |
| 73 automated tests (`uv run pytest`; 83 was the count while the structural/recursive chunker tests existed) | |

## Branches and commits

Branches: `main` (winning path only) and one branch per technique and model, each with a `CHUNKING.md`, the raw eval lines in `eval/runs/` and a README that leads with the technique (E29):
`chunking/structural-{e5-small,bge-m3}`, `chunking/semantic-{e5-small,bge-m3}`, `chunking/structure-semantic-{e5-small,bge-m3}` (semantic + header cuts, not the hybrid), `chunking/structure-then-semantic-{e5-small,bge-m3}` (community two-stage), `chunking/structure-semantic-merge-{e5-small,bge-m3}` (the user's hybrid), The recursive baseline (E20) has no branch: `chunking/recursive-e5-small` (tip `66b0c8f`) was deleted. Retrieval techniques (E30): `retrieval/vector`, `retrieval/keyword`, `retrieval/hybrid`, and the two ES/EN RRF variants that did best, `retrieval/hybrid-es-en-rrf` and `retrieval/vector-es-en-rrf`. The old `chunking/structure-then-semantic` (tip `48861e3`) was deleted; its module and test live in the two two-stage branches.
Main commits in order: `c499977` scaffold, `b9edd96` parse, `cb7adda` chunk, `6373e84` embed, `8c40b66` ingest, `4f5fb05` refactor, `6de9c0a` README, `0a087c0` retrieval,
`24d02f4` docs, `8655b7e` golden set, `3c4bb27` keyword df cap + search_many, `bc8b097` report + terms set, `5cbf826` results, `8fcbbdf` chunker injection, `3bd7110` semantic chunker,
`0846a24` docs, `f5afaa8` structure-first option, `033f9af` validation + grid docs, `f912c1c` sentence-aware splitter, `bfc178c` courses 2-5 set,
`4809081` full-course docs, `46bca68` recursive chunker, `a9e0e3b` docs, `b97c5e5` cut-cause docs, `29f0f82` configurable embedder, `b4ac48b` translations.

## Open experiments and decisions

1. A run that cuts with one model and embeds with the other. (The ES+EN fusion and weighted fusion on the bge-m3 semantic + header cuts index were done in E30.)
2. Decided (E29, E30): bge-m3 + semantic with header cuts + hybrid with ES/EN RRF 1:1, which is what `main` implements; the other techniques stay in their branches. Still open: make the full-course evaluation (prefix of the course 1 sets, ES/EN fusion) reproducible from the repository.
3. Done differently: no `embedding/*` branches (deleted, tips `b05e36f` and `5fd2ac0`; they only held `EMBEDDING.md`). The hardware cost of each model (E24) is a `HARDWARE_<model>.md` in every chunking branch that uses it, and each technique has its `CHUNKING.md` and README.
4. Done: `MAX_DOC_FREQUENCY` is 0.05. Still open: store the embedding model name in the database.
5. Re-measure the reranker ceiling on the full index; fixed-size and clustering chunkers were never tested; int8 vs full precision never compared.
6. MCP server (`mcp/`) and draw.io MCP (`mcp-drawio/`).
