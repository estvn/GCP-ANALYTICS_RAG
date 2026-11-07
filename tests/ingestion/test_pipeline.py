import hashlib

import numpy as np
import pytest

from rag.ingestion.pipeline import ingest
from rag.ingestion.structural import StructuralChunker


class FakeEmbedder:
    """Deterministic 384-d unit vectors; records how many texts it was asked to embed."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += len(texts)
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
            v = np.random.default_rng(seed).normal(size=384)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float32)


class Boom(FakeEmbedder):
    def embed(self, texts, kind):
        raise RuntimeError("embedder down")


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "a.md").write_text("# A\n\nBigQuery is a cloud data warehouse.", encoding="utf-8")
    (tmp_path / "m1" / "b.md").write_text("# B\n\nStorage buckets hold objects.", encoding="utf-8")
    return tmp_path


def texts(conn):
    return {r[0] for r in conn.execute("SELECT text FROM chunks")}


def test_first_ingest_stores_chunks_with_384d_vectors(conn, docs):
    stats = ingest(docs, conn, FakeEmbedder())

    assert stats == {"added": 2, "updated": 0, "removed": 0, "skipped": 0}
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
    assert conn.execute("SELECT vector_dims(embedding) FROM chunks LIMIT 1").fetchone()[0] == 384


def test_full_text_search_column_is_populated(conn, docs):
    ingest(docs, conn, FakeEmbedder())

    n = conn.execute("SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('simple', 'warehouse')").fetchone()[0]
    assert n == 1


def test_unchanged_files_are_not_embedded_again(conn, docs):
    ingest(docs, conn, FakeEmbedder())
    emb = FakeEmbedder()

    stats = ingest(docs, conn, emb)

    assert emb.calls == 0
    assert stats["skipped"] == 2 and stats["added"] == 0


def test_changed_file_replaces_only_its_chunks(conn, docs):
    ingest(docs, conn, FakeEmbedder())
    (docs / "m1" / "a.md").write_text("# A\n\nBigQuery now has new features.", encoding="utf-8")
    emb = FakeEmbedder()

    stats = ingest(docs, conn, emb)

    assert stats["updated"] == 1 and stats["skipped"] == 1
    assert emb.calls == 1
    t = texts(conn)
    assert any("new features" in x for x in t)
    assert not any("cloud data warehouse" in x for x in t)
    assert any("Storage buckets" in x for x in t)


def test_removed_file_loses_its_chunks(conn, docs):
    ingest(docs, conn, FakeEmbedder())
    (docs / "m1" / "b.md").unlink()

    stats = ingest(docs, conn, FakeEmbedder())

    assert stats["removed"] == 1
    assert not any("Storage buckets" in x for x in texts(conn))
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_failed_embedding_keeps_previous_data(conn, docs):
    ingest(docs, conn, FakeEmbedder())
    (docs / "m1" / "a.md").write_text("# A\n\nrewritten", encoding="utf-8")

    with pytest.raises(RuntimeError):
        ingest(docs, conn, Boom())

    assert any("cloud data warehouse" in x for x in texts(conn))


def test_changing_the_chunker_configuration_reindexes_everything(conn, docs):
    ingest(docs, conn, FakeEmbedder(), chunker=StructuralChunker(chunk_size=2400))

    stats = ingest(docs, conn, FakeEmbedder(), chunker=StructuralChunker(chunk_size=1200))

    assert stats["updated"] == 2 and stats["skipped"] == 0


def test_the_default_chunker_is_the_structural_one(conn, docs):
    ingest(docs, conn, FakeEmbedder())

    stats = ingest(docs, conn, FakeEmbedder(), chunker=StructuralChunker())

    assert stats["skipped"] == 2


def test_any_object_with_chunk_and_fingerprint_can_chunk(conn, docs):
    class OneChunkPerFile:
        fingerprint = "one-per-file"

        def chunk(self, text, source):
            return [{"text": text, "source": source, "headers": [], "index": 0}]

    ingest(docs, conn, FakeEmbedder(), chunker=OneChunkPerFile())

    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
