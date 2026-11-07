# 2. Técnicas de retrieval (recuperación)

Guía de estudio. Explica cómo encontramos los chunks que responden una pregunta: la búsqueda vectorial, la búsqueda por
palabras, la fusión RRF, la búsqueda en varios idiomas, qué pasa si se consulta con un modelo que no corresponde, y qué
técnicas no usamos. Las cifras salen de `eval/experiments.md`.

## 1. El flujo de una consulta

```
pregunta --> modelo de embeddings --> vector de la pregunta
         --> busqueda vectorial    --> lista A (30 candidatos ordenados)
         --> busqueda por palabras --> lista B (30 candidatos ordenados)
         --> fusion RRF (A + B)    --> los mejores k chunks (con distancia y origen)
```

No es una sola técnica sino varias piezas con funciones distintas:

| Pieza | Qué es | ¿Usa modelo de embeddings? |
|---|---|---|
| Búsqueda vectorial | Encuentra por significado | **Sí**, es su motor |
| Búsqueda por palabras | Encuentra por coincidencia de palabras | No |
| Fusión RRF | Combina listas ya ordenadas | No |
| Varias formulaciones (complemento) | Consulta la misma pregunta en español e inglés y fusiona | Usa el de la búsqueda vectorial |
| Reranker | Reordena candidatos (no aplicado) | Otro modelo, distinto |

## 2. Búsqueda vectorial

1. La pregunta se convierte en un vector con el mismo modelo que indexó los chunks.
2. La base (pgvector) mide la **distancia coseno** entre ese vector y los vectores de los chunks (0 = idénticos, 2 = opuestos) y
   devuelve los 30 más cercanos.
3. Para no comparar contra todos, usa un índice **HNSW**: un grafo en capas donde se entra por una capa con pocos puntos, se salta
   al vecino más cercano y se baja de capa hasta llegar a la zona correcta. Es aproximado. Parámetros: `m = 16`,
   `ef_construction = 64`, `ef_search = 100` (más alto = más recall y más lento). Con unos 1400 chunks la diferencia con una
   comparación exacta casi no se nota; importa a escala.

**Fortaleza**: entiende sinónimos y paráfrasis ("pago por uso" encuentra "pay as you go"). **Debilidad**: siglas y nombres exactos.
Con `e5-small`, en consultas de términos exactos del índice completo acertó 0.600 (contra 0.900 del híbrido).

## 3. Búsqueda por palabras (full-text)

Cada chunk guarda una columna `tsv` con sus palabras (configuración `simple`, sin stemming porque el corpus mezcla español e
inglés) y un índice **GIN** que funciona como el índice de un libro: cada palabra apunta a los chunks donde aparece.

1. La pregunta se parte en palabras de 3 o más letras, unidas con OR (basta una coincidencia).
2. Se descartan las palabras demasiado frecuentes (ver abajo).
3. El índice devuelve los chunks que contienen alguna; se ordenan con `ts_rank_cd` (cuántas veces y qué tan juntas aparecen).

**No es BM25.** El ranking de texto de Postgres no pondera por rareza del término (IDF): "para" o "sirve" pesaban igual que
"BigQuery". Se aproximó esa ponderación con un filtro: se descartan las palabras que aparecen en más del 15% de los chunks
(y en al menos 5). El 15% se ajustó con 162 chunks; con 750 a 1450 chunks rinde mejor **5%** (de +0.7 a +2.3 puntos), pero el
código todavía usa 15%. Una lista fija de palabras vacías resultó inconsistente. Una extensión con BM25 real
(pg_textsearch, ParadeDB) no se probó.

**Fortaleza**: siglas y nombres exactos (10 de 10 en las consultas de términos del curso 1). **Debilidad**: no entiende
sinónimos ni cruza idiomas, porque busca la misma palabra.

## 4. RRF y la búsqueda híbrida

**RRF (Reciprocal Rank Fusion)** combina listas cuyos puntajes no son comparables (distancia coseno por un lado, ranking de
texto por otro) usando solo las **posiciones**: cada chunk suma `1 / (60 + posición)` por cada lista donde aparece, y gana el de
mayor suma.

```
Chunk A: 1o en vectorial, 3o en palabras -> 1/61 + 1/63 = 0.0323
Chunk B: 2o en vectorial, ausente en palabras -> 1/62      = 0.0161
=> A queda primero
```

- Cada búsqueda aporta 30 candidatos; la fusión junta hasta 60 chunks distintos y devuelve los mejores **k**.
- **RRF no busca nada por sí sola** y con una sola lista no cambia nada: necesita al menos dos.
- **k** no es fijo: se eligió para que cada consulta devuelva unos 9000 caracteres (por ejemplo, 6 chunks de 1500 caracteres o 10
  de 900), así las técnicas de chunking se comparan con la misma cantidad de texto.
- El híbrido existe porque cada búsqueda cubre el punto ciego de la otra.

Resultados (índice completo, 130 preguntas, chunking estructural; aciertos / MRR):

| Búsqueda | `e5-small` | `bge-m3` |
|---|---|---|
| Solo vectorial | 0.662 / 0.489 | 0.785 / 0.599 |
| Híbrido (vectorial + palabras, RRF) | 0.731 / 0.565 | 0.808 / 0.609 |

En las preguntas de los cursos 2 a 5 la diferencia es mayor (`e5-small`: 0.775 solo vectores contra 0.850 a 0.900 con el híbrido, según el umbral de frecuencia). Una corrección importante del
proceso: en el curso 1 el híbrido **no** superó a los vectores solos en preguntas parafraseadas al validar con un conjunto
nuevo; su valor está en siglas y nombres exactos y en el índice grande.

## 5. Varias formulaciones: buscar en español e inglés

El corpus mezcla español e inglés, y el modelo `e5-small` casi no cruza de un idioma a otro (preguntas que cruzan idiomas:
0.09 a 0.23 de aciertos; dentro del mismo idioma: unos 0.87). La solución barata es consultar **la misma pregunta en los dos
idiomas** y fusionar las listas con RRF (`HybridRetriever.search_many`). La traducción la haría el LLM que consulta (Claude, vía
MCP); en las pruebas la hizo un modelo como sustituto.

**RRF por sí misma no es multiidioma**: lo es este uso concreto (dos consultas, una por idioma, fusionadas).

| Índice y búsqueda | Una formulación | RRF 1:1 | RRF 2:1 | RRF 3:1 |
|---|---|---|---|---|
| `e5-small`, estructura + semántica, híbrido: total (mismo idioma / cruce) | 0.762 (0.870 / 0.227) | **0.800** (0.787 / **0.864**) | 0.738 (0.833 / 0.273) | 0.738 (0.833 / 0.273) |
| `bge-m3`, estructural, vectorial | 0.785 (0.787 / 0.773) | 0.792 (0.787 / 0.818) | **0.808** (0.806 / 0.818) | 0.800 (0.796 / 0.818) |
| `bge-m3`, estructural, híbrido | 0.808 (0.880 / 0.455) | **0.846** (0.843 / 0.864) | 0.838 (0.843 / 0.818) | 0.838 (0.843 / 0.818) |

(Pesos = pregunta original : traducción.)

- **Con `e5-small`, solo el reparto 1:1 funciona**: si se da más peso a la pregunta original, el acierto en el mismo idioma se
  recupera pero la mejora entre idiomas se desploma (0.864 a 0.273), porque la consulta traducida es la única que encuentra esas
  respuestas.
- Fusionar los dos idiomas tiene un costo: la consulta en el idioma equivocado mete ruido y en preguntas del mismo idioma se
  pierden de 5 a 21 puntos con `e5-small`.
- **Con `bge-m3`** el modelo ya cruza idiomas por sí solo (0.773 con una sola consulta, contra 0.136 de `e5-small`); la traducción
  suma unos 3 puntos y los pesos casi no importan. Con `bge-m3`, una segunda formulación es opcional.
- Con una sola formulación, el híbrido **perjudica** el cruce de idiomas con `bge-m3` (0.455 contra 0.773 de solo vectores):
  la búsqueda por palabras solo coincide dentro de un idioma y mete ruido en la fusión.

## 6. ¿Qué pasa si se consulta con un modelo que no corresponde al que generó los vectores?

Cada modelo inventa **su propio sistema de coordenadas**. Los números de `e5-small` solo significan algo respecto a cómo ese
modelo ordenó sus 384 dimensiones; los de `bge-m3` respecto a sus 1024. Es como mezclar latitud y longitud con una cuadrícula de
calles. Dos casos, medidos en el índice real de `e5-small` (130 preguntas):

| Caso | Qué ocurre |
|---|---|
| Otra **dimensión** (un vector de 1024 contra un índice de 384) | La base **da error**: "different halfvec dimensions 384 and 1024". Falla de forma visible |
| Misma dimensión pero **otro espacio** (simulado rotando los vectores al azar; no es un segundo modelo real) | **No hay error**, y los resultados son basura: aciertos de 0.662 a **0.000**, peor que adivinar (0.008) |

El segundo caso es el peligroso: el sistema funciona sin quejarse y responde mal. Puede ocurrir al cambiar de modelo sin
reindexar, o al mezclar modelos de la misma dimensión. Para el modelo `e5` además son obligatorios los prefijos `query: ` y
`passage: `, y `bge-m3` no los usa; cada uno debe usarse como lo define su modelo.

**Pendiente (todavía no implementado)**: guardar en la base el nombre del modelo que hizo sus vectores y comprobarlo antes de
buscar, para que el error silencioso no pueda pasar.

Matiz: el modelo que decide **dónde cortar** (chunking semántico) y el que **busca** son independientes, porque lo que sale del
chunking es texto plano. La regla estricta es solo entre los vectores guardados y el vector de la pregunta.

## 7. Reranker (no aplicado)

Un **reranker** es un segundo modelo (cross-encoder) que lee la pregunta y un chunk **juntos** y da un puntaje de relevancia.
Es más preciso que los embeddings (que resumen cada texto por separado) pero lento: hay que correr el modelo por cada par,
así que se aplica a pocos candidatos (por ejemplo, los 30 de la búsqueda híbrida, para quedarse con 5).

Se descartó por medición en el curso 1: aun con 30 candidatos la respuesta estaba presente en el 78-80% de las preguntas (contra
70% en los 5 primeros), así que el techo era de 8 a 10 puntos, y casi todos los fallos eran de cruce de idiomas, que un reranker
no arregla. Salvedad: **no se repitió con el índice completo**, y con `bge-m3` el panorama cambió, así que conviene volver a medirlo.

## 8. Técnicas que no aplicamos

MMR (diversidad de resultados), filtros por metadata, multi-query generada automáticamente por un modelo, parent-child (buscar con
chunks chicos y devolver la sección), HyDE (buscar con una respuesta hipotética), reescritura de la pregunta, grafos y
autoevaluación. Quedaron fuera para no sobrecargar el sistema; se evalúan solo si los datos lo piden.

## 9. Resumen para decidir

| Necesidad | Técnica |
|---|---|
| Preguntas con otras palabras | Búsqueda vectorial |
| Siglas, nombres de servicios, términos exactos | Búsqueda por palabras |
| Cubrir las dos cosas | Híbrido con RRF (más robusto) |
| Corpus en dos idiomas con `e5-small` | Híbrido + RRF 1:1 sobre la pregunta en los dos idiomas |
| Corpus en dos idiomas con `bge-m3` | Una sola formulación basta; la segunda suma unos 3 puntos |
| Más precisión en el orden de los resultados | Reranker (medir de nuevo antes de decidir) |
