import numpy as np

from rag.ingestion.semantic import SemanticChunker, split_units

TOPICS = ["cloud", "pasta", "music"]


class TopicEmbedder:
    """Embeds by topic words; token count is the word count."""

    def embed(self, texts, kind):
        rows = []
        for t in texts:
            v = np.full(len(TOPICS) + 1, 0.02, dtype=np.float32)
            for i, topic in enumerate(TOPICS):
                v[i] += t.lower().count(topic)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)

    def count_tokens(self, text):
        return len(text.split())


def chunker(**kw):
    params = {"max_tokens": 200, "min_tokens": 1, "percentile": 80, **kw}
    return SemanticChunker(TopicEmbedder(), **params)


CLOUD_THEN_PASTA = (
    "# Doc\n\nThe cloud stores data. The cloud scales fast. Cloud costs are metered. "
    "The cloud has regions. Boil the pasta well. Salt the pasta water. "
    "Serve the pasta hot. Fresh pasta cooks fast.\n"
)


def test_a_change_of_topic_starts_a_new_chunk():
    chunks = chunker().chunk(CLOUD_THEN_PASTA, "m/doc.md")

    assert len(chunks) == 2
    assert "cloud" in chunks[0]["text"] and "pasta" not in chunks[0]["text"]
    assert "pasta" in chunks[1]["text"] and "cloud" not in chunks[1]["text"]


def test_one_topic_stays_in_one_chunk():
    text = "# Doc\n\n" + " ".join(f"The cloud fact {i} is true." for i in range(8))

    assert len(chunker().chunk(text, "m/doc.md")) == 1


def test_chunks_never_exceed_the_token_cap():
    text = "# Doc\n\n" + " ".join(f"The cloud fact number {i} is true." for i in range(60))

    chunks = chunker(max_tokens=30).chunk(text, "m/doc.md")

    assert len(chunks) > 3
    body_tokens = [len(c["text"].split("\n", 1)[1].split()) for c in chunks]
    assert max(body_tokens) <= 30


def test_tiny_segments_are_merged_up_to_the_minimum():
    alternating = "# Doc\n\nCloud one. Pasta one. Music one. Cloud two. Pasta two. Music two.\n"

    chunks = chunker(min_tokens=6).chunk(alternating, "m/doc.md")

    assert all(len(c["text"].split("\n", 1)[1].split()) >= 6 for c in chunks[:-1])


def test_chunk_keeps_context_prefix_headers_and_index():
    chunks = chunker().chunk("# Doc\n\n## Part\n\nThe cloud stores data.\n", "m/doc.md")

    assert chunks[0]["text"].startswith("m/doc > Doc > Part\n")
    assert chunks[0]["headers"] == ["Doc", "Part"]
    assert chunks[0]["source"] == "m/doc.md" and chunks[0]["index"] == 0


def test_empty_text_gives_no_chunks():
    assert chunker().chunk("  \n", "m/empty.md") == []


def test_one_word_longer_than_the_cap_is_still_split():
    chunks = chunker(max_tokens=5).chunk("# Doc\n\n" + "cloud " * 40, "m/doc.md")

    assert len(chunks) > 1


def test_units_are_sentences_bullets_and_wrapped_lines():
    text = "# Title\n\nFirst sentence. Second one\nwrapped here.\n\n- bullet a\n- bullet b\n\n¿Pregunta? Sí.\n"

    units = [u.text for u in split_units(text)]

    assert units == [
        "# Title\nFirst sentence.",
        "Second one wrapped here.",
        "- bullet a",
        "- bullet b",
        "¿Pregunta?",
        "Sí.",
    ]


def test_units_remember_the_header_path():
    units = split_units("# A\n\n## B\n\ntext one.\n\n## C\n\ntext two.\n")

    assert [u.headers for u in units] == [("A", "B"), ("A", "C")]


TWO_SECTIONS_SAME_TOPIC = (
    "# A\n\nThe cloud stores data. The cloud scales fast. The cloud is metered.\n\n"
    "# B\n\nThe cloud has regions. The cloud has zones. The cloud is global.\n"
)


def test_headers_are_cut_points_when_structure_comes_first():
    by_meaning = chunker(min_tokens=5, split_on_headers=False).chunk(TWO_SECTIONS_SAME_TOPIC, "m/doc.md")
    by_structure = chunker(min_tokens=5, split_on_headers=True).chunk(TWO_SECTIONS_SAME_TOPIC, "m/doc.md")

    assert len(by_meaning) == 1
    assert [c["headers"] for c in by_structure] == [["A"], ["B"]]


def test_a_tiny_section_is_merged_into_the_next_one_even_with_header_cuts():
    text = "# A\n\nTiny.\n\n# B\n\nThe cloud has regions. The cloud has zones. The cloud is global.\n"

    chunks = chunker(min_tokens=6, split_on_headers=True).chunk(text, "m/doc.md")

    assert len(chunks) == 1


def test_the_fingerprint_changes_with_the_structure_flag():
    assert chunker(split_on_headers=False).fingerprint != chunker(split_on_headers=True).fingerprint
