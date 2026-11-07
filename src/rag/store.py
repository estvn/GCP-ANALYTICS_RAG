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


EF_SEARCH = 100  # HNSW recall/latency knob; starting point from the pgvector skill
MIN_WORD_LENGTH = 3  # shorter words ("es", "de", "to") only add noise to the keyword leg


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
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(EF_SEARCH, limit)),))
        rows = conn.execute(
            "SELECT id, source, idx, headers, text, embedding <=> %s::halfvec(384) AS distance "
            "FROM chunks ORDER BY embedding <=> %s::halfvec(384) LIMIT %s",
            (vector, vector, limit),
        ).fetchall()
    conn.commit()
    return [_hit(row, distance=row[5]) for row in rows]


def keyword_search(conn: psycopg.Connection, query: str, limit: int) -> list[dict]:
    """Chunks sharing at least one word with the query, ranked by ts_rank_cd.

    Not true BM25: Postgres full-text ranking has no IDF weighting.
    """
    words = re.findall(rf"\w{{{MIN_WORD_LENGTH},}}", query.lower())
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
