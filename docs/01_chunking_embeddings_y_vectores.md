# 1. Chunking, modelos de embeddings, vectores y dimensiones

Guía de estudio. Explica cómo se indexa un documento, qué técnicas de chunking usamos, qué hace un modelo de
embeddings, qué son los vectores y sus dimensiones, y por qué hay que usar el mismo modelo al buscar.
Todas las cifras salen de mediciones de este proyecto (ver `eval/experiments.md`); donde algo no se midió, se dice.

## 1. La indexación, paso a paso

Indexar es dejar los documentos preparados para poder buscarlos rápido. Se hace una vez y solo se repite para los
archivos que cambian (el pipeline guarda un hash del texto y de la configuración).

```
PDF/MD --> [1] Markdown --> [2] chunks --> [3] modelo de embeddings --> vector por chunk
       --> [4] Postgres guarda texto + vector + palabras --> [5] Postgres arma los indices
```

| Paso | Qué ocurre | Quién lo hace |
|---|---|---|
| 1. A Markdown | Convierte cada PDF a texto con estructura | Librería `pymupdf4llm` (motor legacy), orquestada por `ingestion/parse.py` |
| 2. Chunks | Corta el texto en trozos | El chunker elegido (sección 2) |
| 3. Embeddings | Convierte cada chunk en una lista de números | `Embedder` (nuestro código), que ejecuta el modelo con `onnxruntime` |
| 4. Guardar | Inserta texto, ruta de headers, vector y columna de palabras | `store.py` (psycopg) hacia Postgres con pgvector |
| 5. Índices | Estructuras para encontrar rápido | **Postgres**, que los construye y mantiene solo; nosotros solo los declaramos una vez |

Los índices son dos: **HNSW** sobre los vectores (búsqueda por significado) y **GIN** sobre las palabras (búsqueda por
texto). Ni el chunking ni los vectores se "indexan" a mano: nosotros calculamos y guardamos, la base organiza.

## 2. Chunking: cortar el texto

Un chunk es un trozo de **texto** (no números). El chunking decide qué trozos existen, y la búsqueda solo puede devolver
lo que el chunking dejó disponible: **el chunking pone el techo de calidad**. Todos los chunkers producen texto; después
cualquiera de ellos sigue el mismo camino de embedding y almacenamiento.

### Tamaño: caracteres, no tokens
Los tamaños del chunking estructural (2400) están en **caracteres**. Un token son unos 4 caracteres, pero la relación
varía: con 2400 caracteres la mediana es 397 tokens y el máximo 951. Importa porque el modelo `e5-small` solo lee 512
tokens por texto (sección 5).

### Las cuatro técnicas (una rama de GitHub cada una)

| Técnica | Cómo corta | ¿Usa modelo? |
|---|---|---|
| **Recursivo solo** (`chunking/recursive-e5-small`) | `RecursiveCharacterTextSplitter` sobre todo el documento: prueba corte en párrafo, línea, fin de oración (`. ? ! ;`), espacio y, al final, carácter; solape de 200. No mira los headers | No |
| **Estructural + recursivo** (`chunking/structural-e5-small`) | 1) corta por headers `#`, `##`, `###`; 2) fusiona secciones vecinas chicas hasta el tamaño límite; 3) las secciones largas se parten con el recursivo y el header queda pegado al primer trozo | No |
| **Semántico** (`chunking/semantic-e5-small`) | Parte el texto en oraciones y viñetas; calcula el embedding de ventanas de 3 unidades; corta donde la distancia entre ventanas vecinas supera el percentil 90 del documento; mínimo 100 tokens, máximo 440 tokens reales | Sí, para decidir dónde cortar |
| **Estructura + semántica** (`chunking/structure-semantic-e5-small`) | Igual que el semántico, y además cada header es un punto de corte (si el chunk ya tiene 100 tokens) | Sí, para decidir dónde cortar |

No se probaron: el **tamaño fijo puro** (cortar cada N caracteres sin mirar nada) y el **agrupamiento por clustering**
(unir piezas parecidas aunque estén lejos en el documento).

### Variantes del chunking semántico

"Semántico" no es una sola técnica. Todas usan embeddings para medir qué tan parecidas son dos piezas de texto vecinas;
cambia la dirección en que se usa esa medida.

| Variante | Dirección | Cómo funciona | Estado en este proyecto |
|---|---|---|---|
| **Corte por quiebre** (top-down) | Parte de oraciones y **corta** | Embebe cada oración con sus vecinas, mide la distancia entre vecinas consecutivas y corta donde supera un umbral (aquí, percentil 90). Es la variante estándar | Probada: `chunking/semantic-e5-small` |
| **Estructura primero, semántica dentro** (híbrido) | Corte por quiebre + títulos | Una sola pasada: cada header `#` es también un punto de corte, y entre headers manda el cambio de significado. El chunk guarda su ruta de títulos | Probada: `chunking/structure-semantic-e5-small` |
| **Fusión** (bottom-up, merge) | Parte de trozos chicos y **fusiona** | 1) se crean trozos chicos (oraciones o párrafos); 2) se embebe cada uno; 3) se recorre el documento y se une el trozo siguiente al actual mientras su similitud supere el umbral y no se pase del tope de tokens; si no, se abre un chunk nuevo | Descartada (decisión del proyecto, no se implementó en `main`; la rama `chunking/semantic-merge` queda como prototipo) |
| **Estructural en dos etapas** | Estructura y luego semántica | 1) se corta por headers en secciones; 2) solo las secciones que superan el tope se subdividen con el corte por quiebre | En prueba: `chunking/structure-then-semantic` (E29) |
| **Clustering** | Agrupa piezas **no vecinas** | Une piezas parecidas aunque estén lejos; rompe el orden de lectura | No probada |

**Qué hace el híbrido hoy.** El código es de una sola pasada. El chunker semántico recibe el documento markdown
crudo (no chunks ya hechos) y los headers entran como cortes extra, siempre que el chunk ya tenga el mínimo de tokens.
Es decir, la estructura marca los cortes gruesos y la semántica los finos, dentro del mismo recorrido. La versión de
dos etapas es distinta: ahí la estructura produce los chunks principales y la semántica solo vuelve a cortar los que
quedaron demasiado grandes.

**Corte contra fusión.** Los dos miden lo mismo (similitud entre piezas vecinas) y cuestan parecido: un embedding por
pieza. La diferencia está en la decisión. El corte compara contra la **distribución de distancias de todo el documento**
(un percentil), así que se adapta a cada texto. La fusión compara contra un **umbral fijo** y contra el chunk que se está
armando, por lo que es más sensible al valor elegido y puede crecer por deriva (cada pieza se parece a la anterior pero
la primera y la última ya no). Las fusiones se descartaron sin medirlas, así que no hay datos propios que digan cuál es más eficiente. La fusión secuencial se parece al método publicado Max-Min (2025), pero la implementación del proyecto es propia y no está validada contra el paper. Lo que sí queda por medir en los
experimento E29 (estructura en dos etapas), con el mismo índice, las mismas preguntas y el mismo texto devuelto.

### Cómo decide el chunking semántico (en lenguaje simple)
Se divide el texto en piezas chicas; el modelo compara cada pieza (con sus vecinas) con la siguiente; si hablan de lo
mismo se mantienen juntas y si el tema cambia se corta. Solo se comparan piezas **vecinas**. Dos reglas lo acotan: un chunk
nunca baja de 100 tokens (se une con el vecino) ni sube de 440 (se corta aunque el tema siga). El tope de tokens es
código propio, medido con el tokenizer del modelo; LangChain solo interviene para el caso raro de una oración más larga
que el tope.

```
1. "BigQuery es un almacen de datos."           | parecidas -> se unen
2. "Guarda y consulta grandes volumenes."       |
3. "Los costos se calculan por bytes leidos."   <- cambia el tema: CORTE
4. "Hay un modo de pago por demanda."           | parecidas -> se unen
5. "Y otro de precio fijo por capacidad."       |
```

En estructura + semántica el reparto medido (muestra de 24 archivos) fue: 47% de los cortes los decide la estructura
(header), 43% la semántica y 10% el tope de tokens; en 33 ocasiones se llegó a un header con el chunk por debajo de
100 tokens y se unió con la sección siguiente.

### Resultados del chunking (índice completo, 311 archivos, 130 preguntas, mismo texto devuelto)

| Técnica | Chunks | Sobre 512 tokens | Aciertos / MRR (mejor variante) |
|---|---|---|---|
| Recursivo solo | 648 | 44% | 0.700 / 0.543 |
| Estructural + recursivo | 755 | 34% | 0.731 / 0.565 |
| Semántico | 1173 | 0.1% | 0.738 / 0.521 |
| Estructura + semántica | 1443 | 0% | **0.762** / 0.562 |

(Con `e5-small`, búsqueda híbrida con umbral de frecuencia 0.05.) El orden en aciertos es consistente, pero las
diferencias (2 a 6 puntos) son una tendencia: el error estándar con 130 preguntas es de unos 4 puntos. Costo: la primera
indexación semántica tarda unas 7 veces más porque calcula un embedding por ventana.

## 3. Embeddings, vectores y coordenadas

Un **modelo de embeddings** convierte un texto en una lista de números. Esa lista se llama **embedding**, **vector** o
**coordenada**: son tres nombres para lo mismo. Se interpreta como la posición del texto en un espacio donde los textos
de significado parecido quedan cerca y los distintos, lejos.

```
chunk (texto) --> modelo de embeddings --> vector de 384 numeros = su coordenada
```

Ejemplo con 3 números (los reales tienen cientos):

```
"BigQuery es un almacen de datos"      -> (0.8, 0.1, 0.2)
"BigQuery guarda grandes volumenes"    -> (0.7, 0.2, 0.2)   cerca del anterior
"Hervir la pasta diez minutos"         -> (0.1, 0.9, 0.7)   lejos
```

Importante:
- El modelo convierte **cada texto por separado**; **no compara chunks entre sí**. Las comparaciones las hace nuestro código
  (entre ventanas vecinas, solo en el chunking semántico) y la base de datos (la pregunta contra los chunks, al buscar).
- Un chunk es texto; lo que tiene 384 o 1024 números es su **vector**.
- Comparar significados es medir la **distancia entre coordenadas** (distancia coseno: cuanto menor, más parecidos).

## 4. Dimensiones

La **dimensión** es cuántos números tiene el vector. **La fija el modelo**, no la base de datos ni nosotros; la tabla se
declara con esa dimensión (`halfvec(384)` o `halfvec(1024)`). Todos los textos, cortos o largos, dan un vector del mismo
tamaño.

| | `multilingual-e5-small` | `bge-m3` |
|---|---|---|
| Dimensión del vector | **384** | **1024** |
| Tokens que lee por texto | 512 | hasta 8192 (usamos 1024) |
| Archivo int8 | unos 118 MB | unos 570 MB |
| Prefijos | `query: ` y `passage: ` (obligatorios) | ninguno |
| Pooling (cómo resume los tokens) | promedio de los tokens | primer token (CLS) |
| Memoria por vector en la base (`halfvec`, 2 bytes por número) | 768 bytes | 2048 bytes |

### Por qué importan más dimensiones
- **Capacidad**: más números permiten distinguir más matices de significado (por ejemplo, separar "pago por demanda" de
  "pago por capacidad" dentro del tema "costos").
- **Costo**: más memoria en la base (aquí 1443 vectores ocupan unos 1.1 MB con 384 dimensiones y unos 3 MB con 1024, algo
  irrelevante a esta escala) y más cálculo en cada comparación.

**Matiz honesto**: la dimensión por sí sola no determina la calidad. `bge-m3` difiere de `e5-small` en tamaño, en
entrenamiento, en dimensiones y en cuántos tokens lee, así que no se puede atribuir su mejora solo a las 1024 dimensiones.
Lo que medimos es el conjunto del modelo:

| Medición en este proyecto | `e5-small` | `bge-m3` |
|---|---|---|
| Aciertos con solo vectores (índice completo, estructural) | 0.662 | 0.785 |
| Aciertos en preguntas que cruzan idiomas (una formulación) | 0.136 | 0.773 |
| RAM en pico al indexar | 912 MB | 1591 MB |
| Velocidad al indexar | 5.6 chunks/s | 0.6 chunks/s (unas 9 veces más lento) |
| Indexar el curso completo (estructural) | unos 2 min | unos 19 min |

Los dos caben en el presupuesto de 20 a 30% de CPU y RAM (con 4 hilos); el precio de `bge-m3` es el tiempo.

## 5. Cuántos tokens lee el modelo y cómo se relaciona con el chunking

`e5-small` lee 512 tokens; lo que pase de ahí no entra en su vector. Con chunks estructurales de 2400 caracteres, el 34% pasa
de 512 tokens. Se midió si eso escondía respuestas: de 130 preguntas, **solo 2 (2%)** tenían su respuesta únicamente en la cola
truncada, y se encontraron igual (la búsqueda por palabras ve el chunk completo). Por eso la truncación no explica que las
técnicas de chunking den resultados parecidos. Los chunkers semánticos garantizan el tope por diseño. Con `bge-m3` (hasta 8192
tokens) los chunks se leen enteros.

## 6. Usar el mismo modelo para indexar y para buscar

Cada modelo inventa **su propio sistema de coordenadas**: los números de un modelo solo significan algo respecto al orden
de dimensiones de ese modelo. Por eso el vector de la pregunta y los vectores guardados deben salir del **mismo modelo**.
Prueba en el índice de `e5-small` (130 preguntas): con un vector de otra dimensión la base da error; con la misma
dimensión pero otro espacio (simulado con una rotación al azar) los aciertos caen de 0.662 a **0.000**. Detalle y
consecuencias en `02_tecnicas_de_retrieval.md`.

Matiz: el modelo que **decide dónde cortar** (chunking semántico) y el que **busca** son independientes, porque lo que sale del
chunking es texto plano. La regla estricta es solo entre los vectores guardados y el vector de la pregunta.

## 7. Cómo elegir (guía práctica)

- Si el presupuesto de tiempo de indexación es corto y todo está en un idioma: `e5-small` rinde bien (0.87 de aciertos en
  preguntas del mismo idioma con híbrido).
- Si hay español e inglés mezclados: `bge-m3` resuelve gran parte del cruce de idiomas con una sola consulta; con `e5-small`
  hace falta consultar en los dos idiomas (ver documento 2).
- Si hay chunks largos (más de 512 tokens): `bge-m3`, o un chunker con tope de tokens.
- Antes de cambiar de modelo: reindexar todo (la base no puede mezclar vectores de modelos distintos).
