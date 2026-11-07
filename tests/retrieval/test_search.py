import numpy as np
import pytest

from rag.ingestion.pipeline import ingest
from rag.retrieval import HybridRetriever
from rag.store import keyword_search, vector_search

TOPICS = {"bigquery": 0, "storage": 1, "pasta": 2}


class TopicEmbedder:
    """Vectors point at the topic words a text mentions, so similarity is predictable."""

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
        "rare": "# Notes\n\nThe zorblax protocol is unrelated to anything else.",
    }
    (tmp_path / "m").mkdir()
    for name, text in docs.items():
        (tmp_path / "m" / f"{name}.md").write_text(text, encoding="utf-8")
    embedder = TopicEmbedder()
    ingest(tmp_path, conn, embedder)
    return HybridRetriever(conn, embedder)


def sources(hits):
    return [h["source"] for h in hits]


def test_semantic_and_keyword_agreement_ranks_first(retriever):
    hits = retriever.search("bigquery data warehouse")

    assert hits[0]["source"] == "m/bq.md"
    assert hits[0]["distance"] is not None and hits[0]["kw_score"] is not None


def test_hit_has_the_fields_the_mcp_needs(retriever):
    hit = retriever.search("bigquery")[0]

    assert {"id", "source", "headers", "text", "score"} <= hit.keys()
    assert hit["headers"] == ["BigQuery"]


def test_rare_exact_term_is_found_through_the_keyword_leg(retriever):
    hits = retriever.search("zorblax", k=4)

    assert "m/rare.md" in sources(hits)


def test_natural_language_question_still_matches_by_keyword(conn, retriever):
    hits = keyword_search(conn, "¿Qué es zorblax?", 5)

    assert sources(hits) == ["m/rare.md"]


def test_k_limits_the_number_of_results(retriever):
    assert len(retriever.search("bigquery storage pasta", k=2)) == 2


def test_hostile_query_does_not_break_the_search(retriever):
    hits = retriever.search("\"; DROP TABLE chunks; -- ' | & ! :* ( ) \\")

    assert isinstance(hits, list)


def test_query_without_words_returns_nothing(retriever):
    assert retriever.search("  ?! ") == []


def test_vector_distance_is_cosine(conn, retriever):
    hits = vector_search(conn, TopicEmbedder().embed(["bigquery"], "query")[0], 4)

    assert 0.0 <= hits[0]["distance"] <= 2.0
    assert hits[0]["source"] == "m/bq.md"
