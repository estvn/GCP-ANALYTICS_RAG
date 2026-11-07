import numpy as np

from rag.ingestion.structure_then_semantic import StructureThenSemanticChunker

TOPICS = ["cloud", "pasta", "music"]


class TopicEmbedder:
    """Embeds by topic words; token count is the word count."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, kind):
        self.calls.append(len(texts))
        rows = []
        for t in texts:
            v = np.full(len(TOPICS) + 1, 0.02, dtype=np.float32)
            for i, topic in enumerate(TOPICS):
                v[i] += t.lower().count(topic)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)

    def count_tokens(self, text):
        return len(text.split())


def chunker(embedder=None, **kw):
    params = {"max_tokens": 30, "min_tokens": 1, "percentile": 70, **kw}
    return StructureThenSemanticChunker(embedder or TopicEmbedder(), **params)


SMALL = "# A\n\nThe cloud stores data.\n\n# B\n\nPasta needs salt."
BIG = "# Big\n\n" + "The cloud stores data. " * 8 + "Pasta needs salt. " * 8


def test_each_small_section_stays_whole_when_min_tokens_is_met():
    chunks = chunker(min_tokens=3).chunk(SMALL, "d.md")
    assert [c["headers"] for c in chunks] == [["A"], ["B"]]


def test_tiny_sections_merge_until_min_tokens():
    assert len(chunker(min_tokens=100, max_tokens=200).chunk(SMALL, "d.md")) == 1


def test_only_oversized_sections_are_cut_by_meaning():
    chunks = chunker().chunk(BIG, "d.md")
    assert len(chunks) >= 2
    assert all(len(c["text"].split()) <= 30 + 4 for c in chunks)
    assert "cloud" in chunks[0]["text"].lower() and "pasta" in chunks[-1]["text"].lower()


def test_small_sections_are_not_embedded_for_cutting():
    embedder = TopicEmbedder()
    chunker(embedder).chunk(SMALL, "d.md")
    assert embedder.calls == []


def test_chunks_carry_source_prefix_and_sequential_index():
    chunks = chunker(min_tokens=3).chunk(SMALL, "dir/d.md")
    assert chunks[0]["text"].startswith("dir/d > A\n") and chunks[0]["source"] == "dir/d.md"
    assert [c["index"] for c in chunks] == [0, 1]


def test_empty_text_gives_no_chunks():
    assert chunker().chunk("", "d.md") == []
