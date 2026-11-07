# GCP Analytics RAG

Question answering over the Google Cloud Data Analytics course (5 courses, about 310 Markdown files, with passages in
both English and Spanish and questions asked in either language). This repository is the **retrieval half** of a RAG system: given a question it
returns the course passages that answer it. It is built to run on a laptop CPU within a small RAM budget (no GPU).

Status: ingestion, hybrid retrieval and evaluation work. The MCP server that exposes it to Claude Code is a separate
repository and is the next step.

## What `main` implements

`main` contains only the techniques that won the experiments. Every other technique lives in its own branch (see
[Branches](#branches)).

```
 course PDFs / Markdown
        │  convert_corpus            pymupdf4llm: PDF -> Markdown
        ▼
   Markdown files
        │  SemanticChunker           cut where the meaning changes, and at every header
        ▼
     chunks (about 880 characters)
        │  bge-m3 (int8, ONNX)       one 1024-number vector per chunk
        ▼
  PostgreSQL + pgvector              vectors (HNSW) and words (full-text index) in one table
        ▲
        │  HybridRetriever.search_many
        │    for each phrasing of the question (original + translation):
        │      vector search + keyword search -> fused with RRF
        │    then the lists of the phrasings are fused with RRF again
        ▼
   top-k chunks
```

- **Chunking: semantic + header cuts.** A chunk ends when the meaning shifts (the embedding distance between
  neighbouring sentences is in the top 10% of the document), when it reaches 440 tokens, or at a `#` header once it
  already has 100 tokens. Each chunk keeps its header path.
- **Embedding model: bge-m3.** Multilingual, so a Spanish question finds English passages. `multilingual-e5-small`
  stays selectable (`load_embedder("e5-small")`): its file is 5 times smaller (113 MB against 542 MB), it peaks at
  about 0.6 times the RAM and it indexes 9 times faster. With the same chunks bge-m3 ranks the correct passage higher and needs about half the text
  for the same coverage (see [Which embedding model](#which-embedding-model-bge-m3-or-e5-small-e33)).
- **Retrieval: hybrid with question + translation fusion.** Vector search finds passages by meaning, keyword search
  finds exact terms, and RRF merges both rankings by position. Running that on the question and on its translation, and
  fusing again, is what rescues questions asked in the other language.

## The winning techniques, and why

Everything in `main` runs these choices by default. Each one links to the experiment that decided it.

| layer | choice in `main` | why it won |
|---|---|---|
| chunking | **semantic + header cuts** (`SemanticChunker`) | It answered the most questions with both models on the 130 original questions (99 with e5-small and 114 with bge-m3, E29) and it returns the least text for the same coverage: 30 chunks are about 26,000 characters, against 38,000 for the author's hybrid chunking and 47,000 for structural chunking (E31). On new questions the chunking techniques are inside the noise of each other (E32), so it was kept as the highest measured with the smallest text. |
| embedding model | **bge-m3** | With this chunker it answers 90% of the 180 questions with 10 chunks, where e5-small needs 17; with 30 chunks it answers 176 and e5-small 165; the correct chunk ranks higher (MRR 0.59 against 0.50) (E36). |
| retrieval | **hybrid + ES/EN 1:1** (`HybridRetriever.search_many`) | The best measured combination: 122 of 130 questions with about 10 chunks, 130 of 130 with 30 chunks, and 46 to 48 of 50 on the new questions (E30, E31, E32). It is described below. |
| chunks returned | **30** | More chunks answer more questions: 163 of 180 with 10, 172 with 20 and 176 with 30 (bge-m3, E36); the price is the text the language model has to read (30 chunks are about 26,000 characters). |

### The retrieval technique: hybrid + ES/EN 1:1

1. **Each phrasing of the question** (the original and its translation) **is searched in two ways**: a vector search (similarity between embeddings, which finds passages by meaning) and a keyword search (PostgreSQL full-text, which finds exact terms), 100 candidates each by default.
2. **Both lists are fused with RRF** (a chunk scores `1 / (60 + position)` in each list it appears in, and the scores add up), which gives one ranking per phrasing.
3. **The rankings of the two phrasings are fused again with RRF**, with equal weight (1:1).
4. **The first 30 chunks of that ranking are returned.**

Why this and not something simpler: keyword search alone is the weakest (it answers 94 to 100 of 130 questions and cannot cross languages); vector search alone has a slightly better order but misses exact terms; fusing both finds more questions; and fusing the translation rescues the questions asked in the other language (with this chunker, from 16 to 21 of the 22 original questions that cross languages, E30). The translation is always fused, with both models.

### If bge-m3 makes the machine slow: switch to e5-small

`main` uses bge-m3 for now. If it proves too heavy, `multilingual-e5-small` is the fallback and the price is known:

- **What is lost:** with this chunker and 30 chunks e5-small answers 165 of 180 questions and bge-m3 176, **11 fewer** (about 6 points); with 20 chunks, 164 against 172. To answer 90% of the questions it needs 17 chunks instead of 10, so the language model reads more text for the same coverage (E36).
- **What is gained:** about 435 MB less memory while loaded (699 MB against 1,135 MB), 4.7 times less CPU time to embed each query (32 against 150 ms), indexing 9 times faster, and a model file of 113 MB instead of 542 MB (E24, E34, E35).
- **If the questions are always in Spanish, the gap between the models is smaller** (E37). 21 of the 96 Spanish questions have their answer in an English passage, so the translation fusion is needed; with it, at 30 chunks, e5-small answers 91 and bge-m3 93 of the 96: **2 questions of difference**, against 11 of 180 when questions in English are included. Without the translation e5-small answers 78 and bge-m3 91, and only 5 against 19 of the 21 questions whose answer is in English.
- **How to switch:** load it with `load_embedder("e5-small")`; create a new database with `init_schema(conn, dim=384)` (the vectors have another length, so an index cannot mix both models) and index again with `ingest` using that embedder; keep the translation fusion; use 30 candidates per search and not 100, because with 100 e5-small needs 23 chunks for 90% (E36).

## Why these choices: the results

All numbers come from the same 130 hand-checked questions (`eval/golden_*.json`): 108 asked in the language of the
passage and 22 asked in the other language. Full ledger: [eval/experiments.md](eval/experiments.md); tables and what
each experiment changed: [eval/results.md](eval/results.md).

### How to read the tables

- **answered** (or *hits*): in how many of the 130 questions a correct passage appears among the `k` returned.
- **hit rate**: the same number as a fraction (114 / 130 = .877).
- **MRR** (mean reciprocal rank): how high the correct passage ranks. Each question scores 1 / position of the first
  correct chunk (1 if first, .5 if second, 0 if absent), averaged over the 130. High hit rate with low MRR means "finds
  it, but ranks it low".
- **k**: how many chunks a query returns. It is not the same for every technique: it is chosen so that every query
  returns about 9,000 characters (`k = 9000 / average chunk length`). Techniques with big chunks get a small `k`
  (6 to 7), techniques with small chunks a large one (10). That way no technique wins just by returning more text.

### Chunking techniques (E29)

Hybrid search, 130 questions. Two embedding models, five chunking techniques.

| chunking technique | model | chunks | k | answered / 130 | hit rate | MRR |
|---|---|---|---|---|---|---|
| structural (headers, 2400 characters) | e5-small | 741 | 6 | 95 | .731 | .561 |
| semantic only | e5-small | 1144 | 9 | 96 | .738 | .517 |
| **semantic + header cuts** | e5-small | 1408 | 10 | **99** | **.762** | .557 |
| community two-stage (headers, then semantic) | e5-small | 1335 | 10 | 97 | .746 | .508 |
| two-stage + merge of neighbours (the author's variant) | e5-small | 907 | 7 | 95 | .731 | .552 |
| structural (headers, 2400 characters) | bge-m3 | 741 | 6 | 107 | .823 | .612 |
| semantic only | bge-m3 | 1128 | 9 | 113 | .869 | .576 |
| **semantic + header cuts** | bge-m3 | 1416 | 10 | **114** | **.877** | .580 |
| community two-stage (headers, then semantic) | bge-m3 | 1336 | 10 | 113 | .869 | .584 |
| two-stage + merge of neighbours (the author's variant) | bge-m3 | 906 | 7 | 109 | .838 | .592 |

- **bge-m3 beats e5-small on every technique** (12 to 17 more questions answered), mostly because of the 22
  questions asked in the other language. This is the one difference in the table that is conclusive.
- **Semantic + header cuts has the highest count with both models** (99 and 114), but its lead over the other chunkers
  is 1 to 7 questions, inside the noise: no chunking technique is conclusively better with the plain hybrid. It is the
  default because it is the highest measured and it also leads in the vector and translation rows of the next table.
- **Community recipe against the author's variant.** The community two-stage recipe (split by headers, subdivide only
  the big sections by meaning) answers 2 to 4 more questions than the same recipe with a merge step, and the merge
  variant has the slightly higher MRR. The gaps (2 and 4 questions) are inside the noise (see the caveats): neither
  beats the other here.
- Semantic + header cuts makes more and smaller chunks than semantic only because of its extra cut at headers.

### Retrieval techniques (E30)

bge-m3, the three best chunking techniques. Cells are *answered / 130* and MRR. `ES/EN` means the question and its
translation were searched and fused with RRF, with weights original:translation of 1:1 or 2:1.

| retrieval | semantic + headers | semantic only | community two-stage |
|---|---|---|---|
| vector only | 112 / .631 | 107 / .581 | 107 / .595 |
| vector + ES/EN 1:1 | 116 / .639 | 104 / .548 | 105 / .594 |
| vector + ES/EN 2:1 | 116 / .638 | 105 / .564 | 106 / .588 |
| keyword only | 75 / .355 | 77 / .374 | 80 / .351 |
| keyword + ES/EN 1:1 | 75 / .300 | 79 / .313 | 73 / .319 |
| keyword + ES/EN 2:1 | 73 / .296 | 75 / .312 | 76 / .320 |
| hybrid (vector + keyword) | 114 / .580 | 113 / .576 | 113 / .584 |
| **hybrid + ES/EN 1:1** | **122** / .591 | 110 / .551 | 114 / .576 |
| hybrid + ES/EN 2:1 | 121 / .607 | 111 / .591 | 112 / .605 |

- **Best measured: hybrid + ES/EN 1:1 on semantic + header cuts** (122 of 130, hit rate .938). Weights 1:1 and 2:1 are
  tied (one question apart); `main` implements 1:1. Its lead is conclusive over semantic only (12 questions) but not
  over two-stage (8 questions) nor over vector + ES/EN (6 questions).
- **Keyword alone is the weakest** (75 to 80 answered): it matches words, not meaning, and cannot cross languages. Its
  job is to be the second leg of the hybrid, for exact terms.
- **Vector alone has the best MRR** (.631 to .639 on semantic + headers) and a simpler pipeline. If ranking the right
  chunk first matters more than finding the most questions, vector + ES/EN is a valid alternative (116 answered).
- **RRF is not a search technique by itself**: it only fuses ranked lists (here `weight / (60 + position)` per list,
  added up). The ES/EN variants apply it on top of vector, keyword and hybrid.
- Translating the question helps only the semantic + headers index; on the other two it is neutral or worse.

### Returning more chunks (E31)

The tables above return about 10 chunks. Returning 30 chunks answers more questions, at three times the text for the language model. Best retrieval technique per bge-m3 index with 30 chunks and a pool of 100 candidates per search (answered / 130 and MRR); details in [eval/experiments.md](eval/experiments.md) E31:

| chunking technique | best retrieval technique | answered / 130 | MRR | characters in 30 chunks |
|---|---|---|---|---|
| semantic + header cuts | hybrid + ES/EN 1:1 | 130 | 0.553 | about 26,356 |
| user's hybrid (two-stage + merge) | hybrid + ES/EN 2:1 | 129 | 0.589 | about 38,052 |
| structural | hybrid + ES/EN 2:1 | 129 | 0.579 | about 47,005 |
| community two-stage | hybrid + ES/EN 2:1 | 125 | 0.591 | about 27,321 |
| semantic only | hybrid + ES/EN 2:1 | 125 | 0.574 | about 31,725 |

- Only semantic + header cuts with hybrid + ES/EN 1:1 answers all 130; the others answer 125 to 129, so the chunking techniques are inside the noise of each other at 30 chunks. Semantic + header cuts returns the least text (about 26,000 characters, against 47,000 for structural).
- At 20 chunks (about 17,600 characters for semantic + header cuts) the best combination answers 127 of 130; the 3 questions it still misses have their correct chunk at positions 22, 24 and 30, so they only appear with more than 20 chunks.
- The 11 questions missed at 10 chunks had their correct chunk between positions 11 and 30: a ranking problem, not a chunking problem (all 130 evidence snippets sit complete inside a chunk of that index).
- A larger pool (100 candidates) helps at 30 chunks and hurts the top 10 and the MRR. The same 130 questions chose chunker, `k` and pool size, so 130 of 130 is optimistic for new questions.

### New questions (E32)

The 130 questions also chose the chunker, `k` and the pool size, so they overstate the result. E32 asks 50 new questions about files that no earlier question used (written by a language model and checked by script against the corpus; details in [eval/experiments.md](eval/experiments.md) E32). Hybrid + ES/EN 1:1, 30 chunks, answered of 130 (30 / 100 candidates) and of the 50 new questions (30 candidates):

| chunking technique | answered of the 130 (30 / 100 candidates) | answered of the 50 new questions at 10, 20, 30 chunks (30 candidates) | at 30 chunks (100 candidates) |
|---|---|---|---|
| semantic + header cuts | 128 / 130 | 41, 46, 48 | 46 |
| structural | 126 / 127 | 45, 48, 50 | 49 |
| user's hybrid (two-stage + merge) | 126 / 127 | 43, 47, 48 | 47 |
| semantic only | 124 / 124 | 39, 45, 46 | 43 |
| community two-stage | 123 / 125 | 42, 48, 48 | 47 |

- The 130 of 130 does not repeat on new questions: semantic + header cuts answers 48 of 50 with 30 candidates and 46 with 100.
- No chunking technique is conclusively better on the 50 new questions (one question is 2 points; under about 6 questions is noise); the structural index answers all 50, semantic + header cuts keeps the smaller text.
- A pool of 100 candidates is not better than 30 on the new questions.
- e5-small catches up with bge-m3 when the translation is fused, so its 5 times smaller file is a serious option; 50 questions cannot separate them.

### Which embedding model: bge-m3 or e5-small (E33)

Both models were run on the same five chunking techniques with the 130 original questions and the 50 new ones (180 in total), 30 candidates per search, with the translation fused (the setting the project always uses). Share of questions answered within the first 10, 20 and 30 chunks, and MRR, averaged over the five chunking techniques ([eval/experiments.md](eval/experiments.md) E33 and the raw output of E31 and E32 in `eval/runs/`):

| retrieval | model | first 10 chunks | first 20 | first 30 | MRR |
|---|---|---|---|---|---|
| hybrid + ES/EN 1:1 | e5-small | 83.4% | 91.7% | 93.6% | 0.49 |
| hybrid + ES/EN 1:1 | bge-m3 | 87.9% | 94.8% | 96.3% | 0.58 |

- **Reading more chunks helps both models**: from 10 to 30 chunks each gains 8 to 10 points. The question is how many chunks each one needs.
- **e5-small needs about twice the chunks.** With the translation fused, e5-small reads 20 chunks (91.7%) to pass what bge-m3 answers with 10 (87.9%); 20 chunks of semantic + header cuts are about 17,600 characters, against about 8,800 for 10.
- **bge-m3 ranks the correct chunk higher.** MRR with the translation: 0.58 against 0.49. This is not an effect of chunking: the chunks are practically the same with both models (741 and 741 structural chunks; 1,408 and 1,416 semantic + header cuts; 907 and 906 for the user's hybrid), and the gap stays on the structural index (structural vector-only on the 130 questions: e5-small answers 107 (MRR 0.50) and bge-m3 124 (MRR 0.62) with the same 741 chunks).
- **Why, from this project's measurements and code**: with a Spanish question against English passages, e5-small scores related and unrelated passages at 0.88 and 0.71 (a gap of 0.17) and bge-m3 at 0.74 and 0.21 (a gap of 0.53), so bge-m3 separates the correct passage better (E24). e5-small also reads only 512 tokens of each chunk and 34% of the structural chunks are longer, and its vectors have 384 numbers against 1,024. That bge-m3 is a much larger model trained for search across many languages is general knowledge, not measured here.
- **The translation is always fused, with both models.** It costs one extra call and two searches per question. It matters much more for e5-small, which answers about 8 points fewer questions at 30 chunks without it, than for bge-m3, where it adds about 1 point (inside the noise); the project fuses it with both models so the pipeline is the same.
- **What bge-m3 costs** (E24): a 542 MB file against 113 MB, a peak of 1,591 MB of RAM against 912 MB, indexing at 0.6 chunks per second against 5.6, and 59 ms per query against 12 ms. The query time is small next to a call to a language model; the indexing time and the RAM are what weigh.
- **Choice for production: bge-m3** (the gap is larger on the chunker `main` uses, see E36), because, with the translation fused, it answers a similar share of questions (the gap is inside the noise) and ranks the correct chunk higher, so the language model gets the same coverage with about half the chunks and the text. e5-small is the option when the RAM is very tight or the whole corpus is re-indexed often; then use 20 to 30 chunks and fuse the translation.
- **Limits**: with 180 questions, differences under about 6 points (about 10 questions) are not conclusive: the gaps between the two models with the translation (2.7 to 4.5 points) are inside the noise: they answer a similar share of questions and differ in the order of the results and in the text they need. Only the position of the correct chunk was measured, not whether a language model answers better with less text; the amount of text is given in characters, not tokens.

### The model that fits the chunker of `main`: semantic + header cuts (E36)

On the chunker `main` uses, with the translation fused, 30 candidates per search and the 180 questions (the 130 original and the 50 new ones), questions answered within the first k chunks ([eval/runs/e36_curve_semh.log](eval/runs/e36_curve_semh.log)):

| chunks returned | e5-small | bge-m3 | difference |
|---|---|---|---|
| 1 | 60 (33%) | **79** (44%) | +19 |
| 3 | 111 (62%) | **121** (67%) | +10 |
| 5 | 127 (71%) | **136** (76%) | +9 |
| 10 | 146 (81%) | **163** (91%) | +17 |
| 15 | 161 (89%) | **167** (93%) | +6 |
| 20 | 164 (91%) | **172** (96%) | +8 |
| 30 | 165 (92%) | **176** (98%) | +11 |

Chunks needed to answer a given share of the 180 questions:

| coverage | chunks e5-small needs | chunks bge-m3 needs |
|---|---|---|
| 85% | 12 | 8 |
| 90% | 17 | 10 |
| 95% | >30 | 18 |

- **bge-m3 answers more with fewer chunks.** To answer 90% of the questions e5-small needs 17 chunks and bge-m3 10; with 10 chunks bge-m3 answers 90.6% against 81.1%. e5-small does not reach 95% even with 30 chunks. MRR: 0.586 against 0.496.
- **With 10 chunks the gap (+17 questions, about 9 points) is larger than the noise of 180 questions (about 6 points); with 15 to 20 chunks it shrinks to 6 to 8 questions, inside the noise.**
- **A pool of 100 candidates widens the gap**: e5-small then needs 23 chunks for 90% and its MRR falls to 0.438, while bge-m3 hardly changes (11 chunks, MRR 0.561).
- **This is among e5-small's weakest combinations** (it ties with the two-stage chunker for the fewest answers). Averaged over the five chunking techniques the gap at 20 chunks is smaller (5.6 questions, E33); on this index e5-small answers 165 of 180 with 30 chunks and bge-m3 176, so the choice of model for this chunker is bge-m3.
- Same limits as E33: only the position of the correct chunk was measured, the 130 original questions chose the settings, and the text is counted in characters.

### What each model costs, and what changes with more chunks (E34)

Measured with the 50 new questions and their translations against the two indexes of the author's hybrid chunking (about 900 chunks each), 4 CPU threads, `search_many` with its defaults (30 chunks, 100 candidates):

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

- **Keeping a model active to answer queries.** With no queries both only hold memory and use no CPU. bge-m3 holds about 435 MB more (1,135 MB against 699 MB, which would be about 860 MB and 445 MB if the second tokenizer were loaded only when needed; the second figure is an estimate) and uses 4.7 times more CPU time to embed each query (150 against 32 ms, a whole `search_many` 424 against 136 ms). The model files are 542 MB and 113 MB on disk. Measured with 4 threads on one machine that was not idle.
- **The vectors are 2.7 times longer, not 5 times** (1,024 against 384 numbers). The 5 times is the model file (542 MB against 113 MB). Because a chunk also stores its text and a full-text index, one chunk takes only 1.4 times more space with bge-m3, not 2.7.
- **What costs time is the model, not the database.** Embedding the query takes 4.7 times longer with bge-m3 and is most of the 424 ms of a search; the two searches in PostgreSQL take about the same with both. The model's time and memory do not grow with the number of chunks.
- **The 1.1 GB stays while the process lives, and it does not grow with queries** (100 queries added 9 MB; it does not depend on the number of chunks). Where it goes (E35):

| step (bge-m3, one process, measured twice with the same result) | memory | added |
|---|---|---|
| Python with the libraries imported | 51 MB | |
| ONNX session loaded (the 568 MB int8 model) | 614 MB | +563 MB |
| first tokenizer (the one that prepares the queries) | 862 MB | +248 MB |
| second tokenizer (the one that only counts tokens for the chunker) | 1,117 MB | +254 MB |
| after 100 different queries | 1,126 MB | +9 MB |

  About 500 MB are the two copies of the tokenizer (the vocabulary has 250,000 tokens), and the second one is only used to count tokens when chunking, which a query never does. Loading it only when needed would leave a query process at about 860 MB with bge-m3; this is not done yet. The same applies to e5-small, which uses the same vocabulary: of its 699 MB, about 500 MB are the two tokenizers and 113 MB the model, so the real difference in memory between the two models is the weights, about 450 MB. Indexing needs more than querying (a peak of 1,591 MB in E24, with batches of 16 chunks).
- **What grows with the number of chunks is storage and indexing time.** Estimate by multiplying the measured bytes per chunk and the indexing speed of E24 (0.6 and 5.6 chunks per second on 60 real chunks); it is an extrapolation, not a measurement:

| chunks | whole table + indexes, bge-m3 / e5-small | vectors + HNSW index only, bge-m3 / e5-small | indexing time, bge-m3 / e5-small |
|---|---|---|---|
| 10,000 | 0.1 GB / 0.1 GB | 0.05 GB / 0.02 GB | 4.6 h / 0.5 h |
| 100,000 | 1.0 GB / 0.7 GB | 0.48 GB / 0.20 GB | 46.3 h / 5.0 h |
| 1,000,000 | 10.0 GB / 7.1 GB | 4.79 GB / 1.96 GB | 463.0 h / 49.6 h |

  The vectors and their HNSW index are 2.4 times larger with bge-m3; HNSW works best when it fits in memory, so at 1,000,000 chunks that is about 4.8 GB against 2.0 GB. Indexing 100,000 chunks would take about two days with bge-m3 on this CPU and five hours with e5-small, unless more threads or a GPU are used.
- **With 310 files there are 900 to 1,400 chunks, not 310**, so at today's size every number above is small: about 9 MB per index and a model in memory.
- **Not measured: whether the quality gap grows with hundreds of thousands of chunks.** With more chunks there are more passages that look alike, and a model that separates the correct passage better (E24) should lose less ranking quality, but this corpus is too small to show it. Testing it needs a larger corpus.
- **A saving on one side can cost on the other.** With e5-small the language model needs about twice the chunks for the same coverage (E33), which is about 8,800 more characters per question (about 2,200 tokens at 4 characters per token, an estimate). With many questions, that text may cost more than the memory saved.

### What the models cost (E24)

Measured on 60 real chunks, 4 threads, int8 ONNX on CPU.

| | e5-small | bge-m3 |
|---|---|---|
| model file | 113 MB | 542 MB |
| peak RAM | 912 MB | 1591 MB |
| indexing speed | 5.6 chunks/s | 0.6 chunks/s |
| query latency (median / p95) | 12 / 18 ms | 59 / 71 ms |

### Caveats

- One run per index and 130 questions. The standard error is about 4 points, so differences under about 8 points
  (about 10 of the 130 questions) are **not conclusive**. The conclusive gaps are: bge-m3 against e5-small (12 to 17
  questions), and semantic + headers against the other two chunkers on vector + ES/EN (11 to 12 questions) and against
  semantic only on hybrid + ES/EN (12 questions).
- The translation of each question is the hand translation in `eval/golden_translations.json`. The translation is
  always fused in production (one extra call and two searches per question). `search_many` takes the phrasings from the
  caller, so a client that can write both versions (such as an LLM) provides them; `search` with a single phrasing is the
  building block of `search_many` and is meant for comparisons.
- The runner scripts that produced the full-course tables are not in this repository (see the reproducibility map in
  [eval/experiments.md](eval/experiments.md)); the raw output is in `eval/runs/`. `main` only has the unweighted
  fusion (1:1); the 2:1 weights are implemented in the branches `retrieval/hybrid-es-en-rrf` and
  `retrieval/vector-es-en-rrf`, whose report (`--translations`) evaluates the same modes.
- The author's hybrid chunking and the e5-small indexes were not run through the retrieval modes of E30.

## Tools and what each is for

| tool | what it does here |
|---|---|
| Python 3.12 + [uv](https://docs.astral.sh/uv/) | runs the code and locks the dependencies (`pyproject.toml`, `uv.lock`) |
| `pymupdf4llm` | converts the course PDFs to Markdown, keeping headers and lists |
| `langchain-text-splitters` | only `RecursiveCharacterTextSplitter`, to split one sentence that is longer than the token cap; the semantic chunker is our own code |
| ONNX Runtime + `tokenizers` | run the embedding models locally on CPU, int8, 4 threads |
| bge-m3 / multilingual-e5-small | turn text into vectors; bge-m3 is the default, e5-small the light option |
| PostgreSQL 17 + pgvector | store chunks, vectors (`halfvec`, HNSW cosine index) and a full-text index (`tsvector`, GIN) in one table |
| `psycopg` | the Postgres driver; only `store.py` imports it |
| Podman | runs the database container (`podman compose` does not work on this machine, so `podman run`) |
| pytest | the tests, which mirror `src/` |

## Steps: from course files to answers

**1. Start the database.**

```bash
podman run -d --name rag-db -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -p 127.0.0.1:5433:5432 -v rag_pgdata:/var/lib/postgresql/data --memory 1g --cpus 2 \
  docker.io/pgvector/pgvector:pg17
```

**2. Install and test.**

```bash
uv sync
uv run pytest
```

**3. Put the model in `models/`** (the folder is git-ignored). bge-m3 needs `onnx/model_int8.onnx`, `tokenizer.json` and
`config.json` in `models/bge-m3/`; the repository of that int8 file is not recorded here. e5-small needs
`onnx/model_quantized.onnx` and `tokenizer.json` from `Xenova/multilingual-e5-small` in `models/multilingual-e5-small/`.

**4. Convert, chunk, embed and store.** Ingestion is incremental: a file is chunked and embedded only when its content
or the chunker settings change, so an interrupted run resumes where it stopped.

```python
from pathlib import Path

import psycopg

from rag.embed import load_embedder
from rag.ingestion import convert_corpus, ingest
from rag.store import init_schema

embedder = load_embedder()                             # bge-m3 by default; load_embedder("e5-small") is the light option
convert_corpus(Path("data/raw"), Path("data/md"))      # PDF -> Markdown

conn = psycopg.connect("postgresql://rag:rag@127.0.0.1:5433/rag")
init_schema(conn, dim=embedder.dim)                    # 1024 for bge-m3, 384 for e5-small
print(ingest("data/md", conn, embedder))               # chunk + embed + store (semantic + header cuts by default)
```

**5. Ask a question.** Give the question and its translation; each is searched (vector + keyword) and the lists are fused.

```python
from rag.retrieval import HybridRetriever
from rag.store import connect

conn = connect("postgresql://rag:rag@127.0.0.1:5433/rag")
retriever = HybridRetriever(conn, embedder)
hits = retriever.search_many(
    ["¿Cómo se particiona una base de datos por columnas?", "How is a database partitioned by columns?"]
)  # 30 chunks, 100 candidates per search
for hit in hits:
    print(hit["source"], hit["headers"], hit["text"][:80])
```

`retriever.search(question)` is the plain hybrid with one phrasing. Both return 30 chunks by default, drawn from a pool of
100 candidates per search (E31); pass `k=10` to return fewer chunks, which means less text for the language model but fewer
questions answered.

**6. Measure.** The golden sets are in `eval/`. The report compares vector, keyword and hybrid; use `--model` to match
the index (`bge-m3` by default, `e5-small` for a 384-number index):

```bash
uv run python -m rag.evaluation.report --model bge-m3
```

The golden sets of course 1 list their sources without the course folder prefix, so the report matches an index built
from that course alone; the full-course tables above used a prefix that the command does not apply.

### Keeping an index

The two indexes of the author's hybrid chunking (`full_STMERGE_bge`: 906 chunks, bge-m3; `full_STMERGE_e5`: 907 chunks,
e5-small; both 309 documents) are dumped to `data/backups/` (git-ignored) and were restored into a temporary database to
check them. Save or restore one like this; each index is queried with its own model (1,024-number vectors for bge-m3,
384 for e5-small):

```bash
podman exec rag-db pg_dump -U rag -Fc full_STMERGE_bge > data/backups/full_STMERGE_bge.dump
podman exec rag-db psql -U rag -d postgres -c 'create database full_STMERGE_bge'
podman exec -i rag-db pg_restore -U rag -d full_STMERGE_bge --no-owner < data/backups/full_STMERGE_bge.dump
```

## Layout

```
src/rag/
├── embed.py          # ONNX embedders: bge-m3 and e5-small (shared by ingestion and retrieval)
├── store.py          # the only module that talks to Postgres
├── ingestion/        # offline pipeline
│   ├── parse.py      # PDF/MD -> markdown
│   ├── semantic.py   # SemanticChunker: semantic cuts + header cuts
│   └── pipeline.py   # incremental ingest (hash-based)
├── retrieval/        # online path
│   ├── fusion.py     # Reciprocal Rank Fusion
│   └── search.py     # HybridRetriever: vector leg + keyword leg, search_many (question + translation)
└── evaluation/       # golden sets, hit rate / MRR, comparison report
eval/                 # golden sets, results.md, experiments.md, runs/ (raw output)
tests/                # mirrors src/
infra/                # container definitions
data/  models/        # local only, git-ignored (course files, ONNX models)
```

## Branches

Each technique that was compared has a branch with a README that explains it and what happened in its experiments.

| technique | branches |
|---|---|
| structural chunking | `chunking/structural-e5-small`, `chunking/structural-bge-m3` |
| semantic chunking | `chunking/semantic-e5-small`, `chunking/semantic-bge-m3` |
| semantic + header cuts (called `structure-semantic`) | `chunking/structure-semantic-e5-small`, `chunking/structure-semantic-bge-m3` |
| community two-stage chunking | `chunking/structure-then-semantic-e5-small`, `chunking/structure-then-semantic-bge-m3` |
| two-stage + merge (the author's variant) | `chunking/structure-semantic-merge-e5-small`, `chunking/structure-semantic-merge-bge-m3` |
| vector, keyword, hybrid search | `retrieval/vector`, `retrieval/keyword`, `retrieval/hybrid` |
| ES/EN fusion | `retrieval/vector-es-en-rrf`, `retrieval/hybrid-es-en-rrf` |

## Study guides (Spanish)

- [docs/01_chunking_embeddings_y_vectores.md](docs/01_chunking_embeddings_y_vectores.md): indexing, chunking techniques, embeddings, vectors and dimensions.
- [docs/02_tecnicas_de_retrieval.md](docs/02_tecnicas_de_retrieval.md): vector search, keyword search, RRF, multi-language fusion, model mismatch.
- [docs/03_arquitecturas_rag.md](docs/03_arquitecturas_rag.md): RAG architectures, what "RAG 1.0 / 2.0" mean.
- [docs/04_este_proyecto.md](docs/04_este_proyecto.md): this project: techniques, tools, advantages and limits.
