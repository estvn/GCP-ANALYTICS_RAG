"""Postgres + pgvector storage. The only module that knows about the database."""

import re

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  source       text PRIMARY KEY,
  content_hash text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source    text NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
  idx       int  NOT NULL,
  headers   text[] NOT NULL,
  text      text NOT NULL,
  embedding halfvec({dim}) NOT NULL,
  -- 'simple' = no stemming: the corpus mixes Spanish and English
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
  UNIQUE (source, idx)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""


def connect(url: str) -> psycopg.Connection:
    """Open a connection that understands the vector types. The schema must already exist."""
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int = 384) -> None:
    conn.execute(SCHEMA.format(dim=int(dim)))
    conn.commit()
    register_vector(conn)


def get_hashes(conn: psycopg.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT source, content_hash FROM documents").fetchall()
    conn.commit()
    return dict(rows)


def replace_document(conn: psycopg.Connection, source: str, content_hash: str, chunks: list[dict], vectors) -> None:
    """Atomically swap all chunks of one document."""
    try:
        conn.execute("DELETE FROM documents WHERE source = %s", (source,))  # cascades to chunks
        conn.execute("INSERT INTO documents (source, content_hash) VALUES (%s, %s)", (source, content_hash))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source, idx, headers, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                [
                    (source, c["index"], c["headers"], c["text"], HalfVector(v.tolist()))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_documents(conn: psycopg.Connection, sources: set[str]) -> None:
    if sources:
        conn.execute("DELETE FROM documents WHERE source = ANY(%s)", (list(sources),))
        conn.commit()


EF_SEARCH = 100  # HNSW recall/latency knob; starting point from the pgvector skill
MIN_WORD_LENGTH = 3  # shorter words ("es", "de", "to") only add noise to the keyword leg
MAX_DOC_FREQUENCY = 0.15  # a word in more than this share of chunks does not discriminate
MIN_COMMON_COUNT = 5  # ...but a tiny corpus cannot tell what is common


def _hit(row, distance=None, kw_score=None) -> dict:
    chunk_id, source, idx, headers, text = row[:5]
    return {
        "id": chunk_id,
        "source": source,
        "idx": idx,
        "headers": headers,
        "text": text,
        "distance": distance,
        "kw_score": kw_score,
    }


def vector_search(conn: psycopg.Connection, query_vector, limit: int) -> list[dict]:
    """Nearest chunks by cosine distance (0 = identical, 2 = opposite)."""
    vector = HalfVector(query_vector.tolist())
    dim = len(query_vector)
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(EF_SEARCH, limit)),))
        rows = conn.execute(
            f"SELECT id, source, idx, headers, text, embedding <=> %s::halfvec({dim}) AS distance "
            f"FROM chunks ORDER BY embedding <=> %s::halfvec({dim}) LIMIT %s",
            (vector, vector, limit),
        ).fetchall()
    conn.commit()
    return [_hit(row, distance=row[5]) for row in rows]


def _discriminating(conn: psycopg.Connection, words: list[str]) -> list[str]:
    """Drop words found in too many chunks. Postgres full-text ranking has no IDF, so we approximate it."""
    if not words:
        return []
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    rows = conn.execute(
        "SELECT w, (SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('simple', w)) "
        "FROM unnest(%s::text[]) AS w",
        (words,),
    ).fetchall()
    limit = max(MAX_DOC_FREQUENCY * total, MIN_COMMON_COUNT)
    return [word for word, frequency in rows if frequency <= limit]


def keyword_search(conn: psycopg.Connection, query: str, limit: int) -> list[dict]:
    """Chunks sharing a discriminating word with the query, ranked by ts_rank_cd.

    Not true BM25. Returns nothing when every query word is too common to be informative.
    """
    words = list(dict.fromkeys(re.findall(rf"\w{{{MIN_WORD_LENGTH},}}", query.lower())))
    words = _discriminating(conn, words)
    conn.commit()
    if not words:
        return []
    rows = conn.execute(
        "SELECT id, source, idx, headers, text, ts_rank_cd(tsv, q) AS rank "
        "FROM chunks, websearch_to_tsquery('simple', %s) AS q "
        "WHERE tsv @@ q ORDER BY rank DESC, id LIMIT %s",
        (" or ".join(words), limit),
    ).fetchall()
    conn.commit()
    return [_hit(row, kw_score=row[5]) for row in rows]
