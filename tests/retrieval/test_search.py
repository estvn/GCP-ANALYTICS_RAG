import numpy as np
import pytest

from rag.ingestion.pipeline import ingest
from rag.retrieval import HybridRetriever
from rag.store import keyword_search, vector_search

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


@pytest.fixture
def crowded(conn, tmp_path):
    """30 chunks all containing 'platform', plus one chunk that also has the rare word."""
    (tmp_path / "m").mkdir()
    for i in range(30):
        (tmp_path / "m" / f"d{i}.md").write_text(f"# Doc {i}\n\nThe platform handles item{i}.", encoding="utf-8")
    (tmp_path / "m" / "rare.md").write_text("# Rare\n\nThe platform also has zorblax.", encoding="utf-8")
    ingest(tmp_path, conn, TopicEmbedder())
    return conn


@pytest.fixture
def widely_used_word(conn, tmp_path):
    """200 chunks; 'widget' is in 12 of them (6%), above the 5% cap but under the old 15% one."""
    (tmp_path / "m").mkdir()
    for i in range(200):
        word = "widget " if i < 12 else ""
        text = f"# Doc {i}\n\n{word}item{i} is here."
        (tmp_path / "m" / f"d{i}.md").write_text(text, encoding="utf-8")
    ingest(tmp_path, conn, TopicEmbedder())
    return conn


def test_a_word_in_more_than_five_percent_of_the_chunks_is_ignored(widely_used_word):
    assert keyword_search(widely_used_word, "widget", 10) == []


def test_words_present_in_most_chunks_are_ignored(crowded):
    hits = keyword_search(crowded, "what does the platform do with zorblax", 10)

    assert sources(hits) == ["m/rare.md"]


def test_query_made_only_of_common_words_finds_nothing(crowded):
    assert keyword_search(crowded, "platform", 10) == []


def test_small_corpus_keeps_every_word(conn, retriever):
    hits = keyword_search(conn, "storage buckets", 5)

    assert sources(hits) == ["m/st.md"]


def test_several_phrasings_are_fused_into_one_ranking(retriever):
    hits = retriever.search_many(["zorblax", "bigquery data warehouse"], k=4)

    assert {"m/rare.md", "m/bq.md"} <= set(sources(hits))


def test_a_chunk_found_by_several_phrasings_appears_once(retriever):
    hits = retriever.search_many(["bigquery", "bigquery data warehouse"], k=10)

    ids = [h["id"] for h in hits]
    assert len(ids) == len(set(ids))


def test_no_phrasings_gives_no_results(retriever):
    assert retriever.search_many([]) == []
