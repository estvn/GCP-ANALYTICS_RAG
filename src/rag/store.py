"""Postgres + pgvector storage. The only module that knows about the database."""

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
  embedding halfvec(384) NOT NULL,
  -- 'simple' = no stemming: the corpus mixes Spanish and English
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
  UNIQUE (source, idx)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
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
