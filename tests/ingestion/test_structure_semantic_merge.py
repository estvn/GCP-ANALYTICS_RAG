import numpy as np

from rag.ingestion.structure_semantic_merge import StructureSemanticMergeChunker

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
    params = {"max_tokens": 40, "min_tokens": 1, "percentile": 50, **kw}
    return StructureSemanticMergeChunker(TopicEmbedder(), **params)


THREE_SECTIONS = (
    "# A\n\nThe cloud stores data.\n\n# B\n\nThe cloud scales fast.\n\n# C\n\nPasta needs salt and water."
)


def test_neighbour_sections_on_the_same_topic_are_merged_across_headers():
    chunks = chunker().chunk(THREE_SECTIONS, "d.md")
    assert len(chunks) == 2
    assert "cloud stores" in chunks[0]["text"] and "cloud scales" in chunks[0]["text"]
    assert chunks[0]["headers"] == ["A"]
    assert "Pasta" in chunks[1]["text"] and "cloud" not in chunks[1]["text"].lower()


def test_a_big_section_is_cut_by_meaning_and_never_exceeds_the_cap():
    big = "# Big\n\n" + "The cloud stores data. " * 8 + "Pasta needs salt. " * 8
    chunks = chunker(max_tokens=30).chunk(big, "d.md")
    assert len(chunks) >= 2
    assert all(len(c["text"].split()) <= 30 + 4 for c in chunks)
    assert "cloud" in chunks[0]["text"].lower() and "pasta" in chunks[-1]["text"].lower()


def test_small_pieces_merge_even_below_the_threshold():
    assert len(chunker(min_tokens=1000, max_tokens=2000, percentile=100).chunk(THREE_SECTIONS, "d.md")) == 1


def test_merging_never_passes_the_token_cap_and_keeps_reading_order():
    chunks = chunker(max_tokens=6, min_tokens=1).chunk(THREE_SECTIONS, "d.md")
    assert all(len(c["text"].split()) <= 6 + 4 for c in chunks)
    body = " ".join(c["text"].split("\n", 1)[1] for c in chunks)
    assert body.index("stores") < body.index("scales") < body.index("Pasta")


def test_chunks_carry_source_prefix_and_sequential_index():
    chunks = chunker().chunk(THREE_SECTIONS, "dir/d.md")
    assert chunks[0]["text"].startswith("dir/d > A\n") and chunks[0]["source"] == "dir/d.md"
    assert [c["index"] for c in chunks] == list(range(len(chunks)))


def test_empty_text_gives_no_chunks():
    assert chunker().chunk("", "d.md") == []
