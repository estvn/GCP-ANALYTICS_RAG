# 4. Este proyecto: técnicas, herramientas y ventajas

Guía de estudio. Resume qué construimos, las técnicas que aplicamos, las herramientas que usamos y por qué, qué resultados
dieron y qué ventajas (y límites) tiene el conjunto. Detalle de cada experimento en `eval/experiments.md`.

## 1. Objetivo

Un RAG de calidad de producción, sin sobreingeniería, sobre el curso **Google Cloud Data Analytics** (5 cursos, 311 archivos, unos
1.8 MB de texto en español e inglés) para que un modelo de lenguaje responda alineado con lo que enseña el curso. Debe correr en
una laptop sin GPU usando entre el 20 y el 30% de CPU y RAM, y es además una pieza de portafolio.

## 2. Arquitectura

Es un **Advanced RAG** con acceso tipo agente (ver `03_arquitecturas_rag.md`):

```
INDEXACION (una vez, incremental por hash)
  PDF/MD -> Markdown -> chunks -> modelo de embeddings -> Postgres + pgvector (texto, vector, palabras, indices)

CONSULTA
  pregunta (en uno o dos idiomas) -> busqueda vectorial + busqueda por palabras -> fusion RRF -> k chunks con su origen
                                  -> el LLM (Claude Code, via un servidor MCP) responde con ellos
```

El servidor MCP que expone la búsqueda a Claude Code **todavía no está construido** (tarea pendiente, en el repositorio
`GCP-ANALYTICS_RAG_MCP`); hoy están hechos la indexación, la recuperación híbrida y la evaluación.

## 3. Técnicas que aplicamos

| Área | Técnica | Detalle |
|---|---|---|
| Parseo | PDF a Markdown con el motor *legacy* de `pymupdf4llm` | El motor de layout rompía las ligaturas ("Certi�cado") |
| Indexación | Incremental por hash de contenido y configuración | Solo se reprocesan los archivos que cambian |
| Chunking | 4 técnicas, una rama cada una | Recursivo solo, estructural + recursivo, semántico, estructura + semántica |
| Embeddings | 2 modelos soportados | `multilingual-e5-small` (384 d) y `bge-m3` (1024 d), int8 en ONNX Runtime, 4 hilos |
| Búsqueda vectorial | Distancia coseno con índice HNSW en pgvector | Vectores en media precisión (`halfvec`) |
| Búsqueda por palabras | Full-text de Postgres con índice GIN | Con filtro de palabras demasiado frecuentes (aproxima el IDF; no es BM25) |
| Fusión | RRF (Reciprocal Rank Fusion), 30 candidatos por búsqueda | `k` ajustado para devolver unos 9000 caracteres |
| Idiomas | Consulta en español e inglés fusionada con RRF (`search_many`) | Esencial con `e5-small`, opcional con `bge-m3` |
| Evaluación | 130 preguntas en 4 conjuntos, `hit@k` y MRR, texto devuelto igual, conjunto de validación aparte | Citas verificadas literalmente contra los archivos |
| Calidad | Pruebas primero (TDD), 83 pruebas automáticas, Bandit y pip-audit antes de publicar | Sin hallazgos hasta ahora |
| Registro | Cada experimento documentado en `eval/experiments.md`, con un registro de correcciones | Incluye lo que salió mal |

## 4. Herramientas

| Herramienta | Para qué | Por qué la elegimos |
|---|---|---|
| Python 3.12 y `uv` | Entorno y dependencias | Rápido y reproducible (la máquina trae 3.14, que aún no soportan todas las librerías) |
| PostgreSQL + **pgvector** 0.8.7 (Podman) | Base vectorial, texto y palabras en un solo lugar | Una sola base para vectores y búsqueda por texto; sin servicios extra |
| `onnxruntime` + `tokenizers` | Ejecutar los modelos de embeddings en CPU | Bajo consumo de RAM y buena velocidad en CPU frente a un framework completo de deep learning |
| `multilingual-e5-small` y `bge-m3` | Convertir texto en vectores | El primero es liviano; el segundo cruza idiomas mucho mejor |
| `pymupdf4llm` | PDF a Markdown | Conserva headers y estructura |
| `langchain-text-splitters` | Cortar texto (solo esa pieza de LangChain) | Cortadores probados; el resto (almacenamiento, búsqueda, fusión) es código propio |
| `psycopg` | Conexión a Postgres | Es el único módulo (`store.py`) que habla con la base |
| `pytest`, Bandit, pip-audit | Pruebas y seguridad | Calidad verificable |
| Podman | Contenedor de la base | `podman compose` no funciona en esta máquina; se usa `podman run` |
| MCP (SDK oficial) | Exponer la búsqueda a Claude Code | Pendiente |
| GitHub | 3 repositorios + ramas por técnica | Cada técnica comparada vive en su rama |

## 5. Resultados principales

Índice completo (311 archivos, 130 preguntas, mismo texto devuelto; aciertos / MRR):

| Combinación | Resultado |
|---|---|
| `e5-small` + estructural + híbrido | 0.731 / 0.565 |
| `e5-small` + estructura + semántica + híbrido | 0.762 / 0.562 |
| `bge-m3` + estructural + solo vectorial | 0.785 / 0.599 |
| `bge-m3` + estructural + híbrido | 0.808 / 0.609 |
| `bge-m3` + estructural + híbrido + dos idiomas (RRF 1:1) | **0.846** / 0.585 |

- Dentro de un mismo idioma el sistema acierta alrededor del 87% (híbrido); entre idiomas, `e5-small` casi no cruza (0.09 a
  0.23) y `bge-m3` sí (0.77 con una sola consulta).
- `bge-m3` mejora todas las variantes (+12 puntos con solo vectores), a cambio de indexar unas 9 veces más lento y usar 1.6 GB de RAM en pico.
- Pendiente: `bge-m3` con chunking de estructura + semántica (indexación en curso).

## 6. Ventajas del conjunto

| Ventaja | Evidencia |
|---|---|
| **Corre en una laptop sin GPU dentro del presupuesto** | CPU 22-23% promedio y RAM de 0.9 a 1.6 GB en pico con 4 hilos |
| **Cubre significado y términos exactos** | El híbrido gana a los vectores solos en las tres técnicas de chunking con el índice completo; con siglas, 0.600 contra 0.900 |
| **Robusto entre idiomas** | `bge-m3` pasa de 0.136 a 0.773 en preguntas que cruzan idiomas; con `e5-small` la fusión en dos idiomas lo sube de 0.23 a 0.86 |
| **Un solo almacén** | Vectores, texto y palabras en Postgres: menos piezas que operar |
| **Indexación incremental** | Cambiar un archivo no reindexa el resto (re-ejecución sin cambios: 0.01 s) |
| **Chunks que respetan el límite del modelo** | Los chunkers semánticos dejan 0 chunks sobre 512 tokens (el estructural: 34%) |
| **Medible y reproducible** | Conjuntos de preguntas con citas verificadas, conjunto de validación separado, comando `report`, registro completo |
| **Privacidad y costo** | Los modelos de recuperación corren localmente: no consumen tokens del LLM (cada consulta entrega unos 3 mil tokens) |
| **Modular** | Chunkers, modelos y búsquedas son intercambiables; cada técnica comparada está en su rama |
| **Honesto con sus límites** | El registro incluye las correcciones y lo que no se midió |

## 7. Límites y riesgos

- **Conjuntos de evaluación chicos**: con 130 preguntas, diferencias menores a unos 4 puntos (error estándar) son ruido; las preguntas las redactó un modelo y solo se acepta una respuesta correcta por pregunta, lo que subestima los aciertos.
- **Tamaño del corpus**: son unos 1.8 MB (del orden de 400 mil tokens, estimación). A esta escala el RAG gana por costo y velocidad más que por calidad frente a pasarle el texto crudo al modelo; esa comparación de extremo a extremo no se hizo.
- **Cruce de idiomas con `e5-small`**: depende de consultar en los dos idiomas (la traducción la haría el LLM que consulta).
- **Sin registro del modelo en la base**: hoy nada impide consultar con un modelo distinto al que indexó (ver `02_tecnicas_de_retrieval.md`).
- **Umbral de frecuencia**: el código usa 15%, pero con índices grandes rinde mejor 5%.
- **Ruido de la conversión de PDF**: números de página y saltos de párrafo dentro de oraciones (22 casos en 2 de 70 archivos).
- **Costo de `bge-m3` con chunking semántico**: la primera indexación puede tardar unas 2 horas (después es incremental).
- **No medido**: reranker con el índice completo, chunker de tamaño fijo, clustering, int8 contra precisión completa.

## 8. Estructura del repositorio y ramas

```
src/rag/
  embed.py          modelos de embeddings (e5-small, bge-m3), pooling y dimension
  store.py          unico modulo que habla con Postgres
  ingestion/        parse, chunk (recursivo y estructural), semantic, pipeline
  retrieval/        fusion (RRF), search (HybridRetriever, search_many)
  evaluation/       golden, metrics, report
tests/  eval/  infra/  docs/        (data/ y models/ fuera de git)
```

Ramas: `main` (todo), `chunking/recursive`, `chunking/structural`, `chunking/semantic`, `chunking/structure-semantic`. La
idea de trabajo: lo que gane en cada comparación queda como único camino en `main`, y las alternativas quedan en sus ramas.

## 9. Cómo ejecutarlo

Base de datos (Podman):

```bash
podman run -d --name rag-db -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -p 127.0.0.1:5433:5432 -v rag_pgdata:/var/lib/postgresql/data --memory 1g --cpus 2 \
  docker.io/pgvector/pgvector:pg17
```

Una base de datos por modelo, porque la tabla fija la dimensión del vector (384 o 1024):

```bash
podman exec rag-db psql -U rag -d postgres -c "CREATE DATABASE rag_bge"
```

Indexar y buscar (API de Python; todavía no hay línea de comandos de ingesta):

```python
import psycopg
from rag.embed import load_embedder
from rag.ingestion import SemanticChunker, ingest
from rag.retrieval import HybridRetriever
from rag.store import init_schema

emb = load_embedder("bge-m3")                                   # o "e5-small" (en otra base)
conn = psycopg.connect("postgresql://rag:rag@127.0.0.1:5433/rag_bge")
init_schema(conn, dim=emb.dim)
ingest("data/md", conn, emb, chunker=SemanticChunker(emb, split_on_headers=True))

r = HybridRetriever(conn, emb)
r.search("¿Qué es BigQuery?", k=6)
r.search_many(["¿Qué es BigQuery?", "What is BigQuery?"], k=6)  # fusion RRF 1:1 en dos idiomas
```

Evaluar y probar:

```bash
uv run python -m rag.evaluation.report --model bge-m3
uv run pytest
```

(La fusión con pesos distintos y las grillas de experimentos se corrieron con scripts temporales que no están en el repositorio.)

## 10. Pendientes

1. Terminar `bge-m3` + estructura + semántica y sus pruebas de fusión en dos idiomas.
2. Decidir la configuración final y dejar en `main` solo ese camino; ramas por modelo de embeddings.
3. Guardar en la base el nombre del modelo; bajar el umbral de frecuencia a 5%.
4. Servidor MCP del RAG y MCP de draw.io, en sus repositorios.
5. Opcionales: reranker con el índice completo, evaluación de extremo a extremo, limpieza de números de página del PDF.
