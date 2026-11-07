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
  about 0.6 times the RAM and it indexes 9 times faster.
- **Retrieval: hybrid with question + translation fusion.** Vector search finds passages by meaning, keyword search
  finds exact terms, and RRF merges both rankings by position. Running that on the question and on its translation, and
  fusing again, is what rescues questions asked in the other language.

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
- The translation of each question is the hand translation in `eval/golden_translations.json`. A real deployment needs
  a translation step: one extra call and two searches per question. `search_many` takes the phrasings from the caller,
  so a client that can write both versions (such as an LLM) provides them; with a single phrasing it is the plain hybrid.
- The runner scripts that produced the full-course tables are not in this repository (see the reproducibility map in
  [eval/experiments.md](eval/experiments.md)); the raw output is in `eval/runs/`. The 2:1 weights were evaluated with
  one of those scripts: `main` only has the unweighted fusion.
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

embedder = load_embedder("bge-m3")
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
    ["¿Cómo se particiona una base de datos por columnas?", "How is a database partitioned by columns?"], k=10
)
for hit in hits:
    print(hit["source"], hit["headers"], hit["text"][:80])
```

`retriever.search(question, k=10)` is the plain hybrid with one phrasing.

**6. Measure.** The golden sets are in `eval/`. The report compares vector, keyword and hybrid; use `--model` to match
the index (`bge-m3` by default, `e5-small` for a 384-number index):

```bash
uv run python -m rag.evaluation.report --model bge-m3
```

The golden sets of course 1 list their sources without the course folder prefix, so the report matches an index built
from that course alone; the full-course tables above used a prefix that the command does not apply.

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
