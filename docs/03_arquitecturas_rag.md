# 3. Arquitecturas de RAG: RAG 1.0, RAG 2.0 y las demás

Guía de estudio. Explica las arquitecturas de RAG, qué significan realmente los nombres "RAG 1.0" y "RAG 2.0", y en cuál
cae este proyecto.

## 1. Qué es RAG

**RAG (Retrieval-Augmented Generation)** es una técnica donde, antes de responder, un sistema **busca** fragmentos relevantes en una
base de conocimiento y se los da a un modelo de lenguaje para que **genere** la respuesta apoyándose en ellos. Así el modelo
puede usar información que no estaba en su entrenamiento (documentos propios, material reciente) y se puede citar de dónde sale.
La idea se atribuye a trabajos de 2020 de Facebook AI Research, en los que participó Douwe Kiela.

## 2. Una advertencia sobre los nombres

Los nombres "RAG 1.0" y "RAG 2.0" **no son términos estandarizados**. Esto es lo que se pudo verificar:

- **"RAG 2.0" es el nombre que usa la empresa Contextual AI** para un enfoque en el que el recuperador y el modelo de lenguaje se
  entrenan **juntos, de extremo a extremo, como un solo sistema**, en lugar de unir piezas ya entrenadas y congeladas. Las cifras
  de mejora que publica la empresa son afirmaciones de su producto y no las verificamos.
- **"RAG 1.0" no es un término formal.** Se usa de manera informal para el RAG "original" o estándar: un recuperador y un modelo de
  lenguaje independientes, ya entrenados, unidos por un pipeline.
- La clasificación **académica** más citada es la del survey de Gao et al. (2023): **Naive RAG, Advanced RAG y Modular RAG**.

Por eso, cuando alguien diga "RAG 1.0 o 2.0", conviene preguntar qué definición usa.

## 3. Las arquitecturas

### 3.1 Naive RAG (el "RAG 1.0" informal)
El pipeline básico, también llamado "recuperar y leer" (retrieve-read):

```
indexar: documentos -> chunks -> embeddings -> base vectorial
consultar: pregunta -> buscar los k chunks mas parecidos -> pegarlos en el prompt -> el LLM responde
```
Es simple y rápido de armar. Sus problemas habituales: recupera chunks poco relevantes o redundantes, falla con términos exactos, no
cruza idiomas si el modelo es débil en eso, y no sabe cuándo la recuperación fue mala.

### 3.2 Advanced RAG
Agrega optimizaciones **antes** y **después** de recuperar para corregir esos problemas:

| Etapa | Técnicas típicas |
|---|---|
| Antes de recuperar | Chunking con criterio (por estructura o por significado), metadata, reescritura de la pregunta, expansión (varias formulaciones) |
| Al recuperar | Búsqueda híbrida (vectorial + por palabras), fusión de rankings (RRF) |
| Después de recuperar | Reranking, compresión o filtrado del contexto |

### 3.3 Modular RAG
Trata el sistema como **módulos intercambiables** (búsqueda, memoria, enrutamiento, adaptadores de tarea, fusión de
consultas) que se combinan según la tarea: por ejemplo, un enrutador decide a qué fuente consultar, o se repite la
búsqueda con otra formulación. Es una generalización del Advanced RAG.

### 3.4 RAG 2.0 (según Contextual AI)
El recuperador y el generador se **entrenan juntos** con el objetivo de responder bien, propagando el aprendizaje por ambos. Requiere
datos de entrenamiento y cómputo, y es un enfoque de producto de esa empresa; no es algo que se arme solo configurando un pipeline.

### 3.5 Otros patrones que aparecen en la literatura y en la industria
**RAG con agentes** (el modelo decide cuándo y qué buscar, y puede hacer varias búsquedas), **Graph RAG** (recuperar sobre un grafo
de entidades y relaciones), **Self-RAG y Corrective RAG** (el sistema evalúa la calidad de lo recuperado y reintenta). No los
verificamos en esta sesión; se mencionan solo para ubicar el mapa.

## 4. Comparación

| Arquitectura | Qué cambia | ¿Requiere entrenar? | Costo de armarla |
|---|---|---|---|
| Naive (RAG 1.0 informal) | Pipeline básico | No | Bajo |
| Advanced | Optimizaciones antes y después de buscar | No | Medio |
| Modular | Módulos y rutas combinables | No (normalmente) | Medio a alto |
| RAG 2.0 (Contextual AI) | Recuperador y generador entrenados juntos | **Sí** | Alto |
| Con agentes | El modelo decide qué y cuándo buscar | No | Medio a alto |

## 5. Dónde cae este proyecto

Es un **Advanced RAG** con acceso tipo agente:

- **Antes de recuperar**: chunking con criterio (estructura + semántica con tope de tokens), metadata (ruta de headers), y varias
  formulaciones de la pregunta (español e inglés).
- **Al recuperar**: búsqueda híbrida (vectorial + palabras) con fusión **RRF**.
- **Después de recuperar**: *no* hay reranker (se descartó por medición, a re-evaluar con `bge-m3`).
- **Generación**: la hace un modelo externo (Claude Code) que llama a una herramienta **MCP**. Como es el modelo quien decide cuándo
  buscar y puede consultar varias veces (por ejemplo, en dos idiomas), el uso es de tipo agente. El servidor MCP **todavía no
  está construido**; es la tarea pendiente.

No es "RAG 2.0": ningún componente se entrena; el modelo de embeddings, el índice y el LLM son piezas independientes y ya entrenadas.
En el sentido informal es una versión mejorada del "RAG 1.0" (piezas congeladas unidas por un pipeline).

## 6. Cuándo conviene cada una

- **Naive**: prototipos, corpus chicos, un solo idioma y preguntas que repiten las palabras del texto.
- **Advanced**: casi todo uso real; es el punto de equilibrio entre calidad y complejidad. Es lo que se aplicó aquí.
- **Modular o con agentes**: varias fuentes, tareas distintas, o cuando el modelo debe decidir qué buscar.
- **RAG 2.0 (entrenado de extremo a extremo)**: solo si hay presupuesto, datos de entrenamiento y un caso donde los componentes
  genéricos no alcanzan.

Para subir de nivel desde este proyecto, los pasos naturales serían: un reranker multilingüe (medir de nuevo su techo con el
índice completo), reescritura de la pregunta hecha por el propio modelo que consulta, y una evaluación de extremo a extremo
(que mida si la respuesta final es correcta, no solo si el fragmento aparece).

## Fuentes
- Contextual AI, "Introducing RAG 2.0": https://contextual.ai/introducing-rag2
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey", arXiv:2312.10997: https://arxiv.org/html/2312.10997v5
