import numpy as np
import pytest

from rag.ingestion.pipeline import ingest
from rag.retrieval import VectorRetriever
from rag.store import vector_search

TOPICS = {"bigquery": 0, "storage": 1, "pasta": 2}


class TopicEmbedder:
    """Vectors point at the topic words a text mentions, so similarity is predictable."""

    def count_tokens(self, text):
        return len(text.split())

    def embed(self, texts, kind):
        rows = []
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            v[100] = 0.3  # keeps the vector non-zero for texts without topics
            for word, dim in TOPICS.items():
                if word in t.lower():
                    v[dim] += 1.0
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)


@pytest.fixture
def retriever(conn, tmp_path):
    docs = {
        "bq": "# BigQuery\n\nBigQuery is a cloud data warehouse for analytics.",
        "st": "# Storage\n\nStorage buckets hold objects.",
        "pasta": "# Pasta\n\nBoil the pasta for ten minutes.",
    }
    (tmp_path / "m").mkdir()
    for name, text in docs.items():
        (tmp_path / "m" / f"{name}.md").write_text(text, encoding="utf-8")
    embedder = TopicEmbedder()
    ingest(tmp_path, conn, embedder)
    return VectorRetriever(conn, embedder)


def sources(hits):
    return [h["source"] for h in hits]


def test_the_closest_chunk_ranks_first(retriever):
    hits = retriever.search("bigquery data warehouse")

    assert hits[0]["source"] == "m/bq.md"


def test_hit_has_the_fields_the_mcp_needs(retriever):
    hit = retriever.search("bigquery")[0]

    assert {"id", "source", "headers", "text", "distance"} <= hit.keys()
    assert hit["headers"] == ["BigQuery"]


def test_k_limits_the_number_of_results(retriever):
    assert len(retriever.search("bigquery storage pasta", k=2)) == 2


def test_hostile_query_does_not_break_the_search(retriever):
    hits = retriever.search("\"; DROP TABLE chunks; -- ' | & ! :* ( ) \\")

    assert isinstance(hits, list)


def test_query_without_words_returns_nothing(retriever):
    assert retriever.search("  ?! ") == []


def test_vector_distance_is_cosine(conn, retriever):
    hits = vector_search(conn, TopicEmbedder().embed(["bigquery"], "query")[0], 3)

    assert 0.0 <= hits[0]["distance"] <= 2.0
    assert hits[0]["source"] == "m/bq.md"


def test_several_phrasings_are_fused_into_one_ranking(retriever):
    hits = retriever.search_many(["bigquery", "pasta"], k=2)

    assert set(sources(hits)) == {"m/bq.md", "m/pasta.md"}


def test_a_chunk_found_by_several_phrasings_appears_once(retriever):
    hits = retriever.search_many(["bigquery", "bigquery data warehouse"], k=10)

    ids = [h["id"] for h in hits]
    assert len(ids) == len(set(ids))


def test_no_phrasings_gives_no_results(retriever):
    assert retriever.search_many([]) == []


def test_the_heavier_phrasing_decides_the_first_hit(retriever):
    assert sources(retriever.search_many(["bigquery", "pasta"], weights=(9, 1)))[0] == "m/bq.md"
    assert sources(retriever.search_many(["bigquery", "pasta"], weights=(1, 9)))[0] == "m/pasta.md"


def test_weights_must_match_the_number_of_phrasings(retriever):
    with pytest.raises(ValueError):
        retriever.search_many(["bigquery", "pasta"], weights=(1,))
