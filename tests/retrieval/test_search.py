import numpy as np
import pytest

from rag.ingestion.pipeline import ingest
from rag.retrieval import KeywordRetriever


class ZeroEmbedder:
    """Ingestion needs vectors, but keyword retrieval ignores them."""

    def count_tokens(self, text):
        return len(text.split())

    def embed(self, texts, kind):
        v = np.zeros(384, dtype=np.float32)
        v[0] = 1.0
        return np.tile(v, (len(texts), 1))


def write_docs(tmp_path, docs):
    (tmp_path / "m").mkdir()
    for name, text in docs.items():
        (tmp_path / "m" / f"{name}.md").write_text(text, encoding="utf-8")


def sources(hits):
    return [h["source"] for h in hits]


@pytest.fixture
def retriever(conn, tmp_path):
    write_docs(
        tmp_path,
        {
            "bq": "# BigQuery\n\nBigQuery is a cloud data warehouse for analytics.",
            "st": "# Storage\n\nStorage buckets hold objects.",
            "rare": "# Notes\n\nThe zorblax protocol is unrelated to anything else.",
        },
    )
    ingest(tmp_path, conn, ZeroEmbedder())
    return KeywordRetriever(conn)


def test_hit_has_the_fields_the_mcp_needs(retriever):
    hit = retriever.search("bigquery")[0]

    assert {"id", "source", "headers", "text", "kw_score"} <= hit.keys()
    assert hit["headers"] == ["BigQuery"]


def test_rare_exact_term_is_found(retriever):
    assert sources(retriever.search("zorblax")) == ["m/rare.md"]


def test_natural_language_question_still_matches_by_keyword(retriever):
    assert sources(retriever.search("¿Qué es zorblax?")) == ["m/rare.md"]


def test_k_limits_the_number_of_results(retriever):
    assert len(retriever.search("bigquery storage zorblax", k=2)) == 2


def test_hostile_query_does_not_break_the_search(retriever):
    hits = retriever.search("\"; DROP TABLE chunks; -- ' | & ! :* ( ) \\")

    assert isinstance(hits, list)


def test_query_without_words_returns_nothing(retriever):
    assert retriever.search("  ?! ") == []


@pytest.fixture
def crowded(conn, tmp_path):
    """30 chunks all containing 'platform', plus one chunk that also has the rare word."""
    docs = {f"d{i}": f"# Doc {i}\n\nThe platform handles item{i}." for i in range(30)}
    docs["rare"] = "# Rare\n\nThe platform also has zorblax."
    write_docs(tmp_path, docs)
    ingest(tmp_path, conn, ZeroEmbedder())
    return KeywordRetriever(conn)


@pytest.fixture
def widely_used_word(conn, tmp_path):
    """200 chunks; 'widget' is in 12 of them (6%), above the 5% cap but under the old 15% one."""
    docs = {f"d{i}": f"# Doc {i}\n\n{'widget ' if i < 12 else ''}item{i} is here." for i in range(200)}
    write_docs(tmp_path, docs)
    ingest(tmp_path, conn, ZeroEmbedder())
    return KeywordRetriever(conn)


def test_a_word_in_more_than_five_percent_of_the_chunks_is_ignored(widely_used_word):
    assert widely_used_word.search("widget", 10) == []


def test_words_present_in_most_chunks_are_ignored(crowded):
    hits = crowded.search("what does the platform do with zorblax", 10)

    assert sources(hits) == ["m/rare.md"]


def test_query_made_only_of_common_words_finds_nothing(crowded):
    assert crowded.search("platform", 10) == []


def test_small_corpus_keeps_every_word(retriever):
    assert sources(retriever.search("storage buckets")) == ["m/st.md"]
