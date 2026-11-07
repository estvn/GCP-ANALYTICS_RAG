from rag.ingestion.chunk import chunk_markdown

SMALL = "# Doc\n\n## A\n\nalpha text.\n\n## B\n\nbeta text.\n\n## C\n\ngamma text.\n"


def test_small_sections_are_merged_up_to_chunk_size():
    chunks = chunk_markdown(SMALL, source="mod/doc.md", chunk_size=500)

    assert len(chunks) == 1
    for word in ("alpha", "beta", "gamma"):
        assert word in chunks[0]["text"]


def test_merge_respects_chunk_size():
    chunks = chunk_markdown(SMALL, source="mod/doc.md", chunk_size=60, overlap=0)

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 60 + len("mod/doc > Doc > A\n") + 20 for c in chunks)


def test_big_section_is_split_with_overlap():
    big = "# Doc\n\n## Long\n\n" + " ".join(f"word{i}" for i in range(400))

    chunks = chunk_markdown(big, source="mod/doc.md", chunk_size=300, overlap=50)

    assert len(chunks) > 3
    assert all(c["headers"] == ["Doc", "Long"] for c in chunks)
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"]


def test_chunks_carry_context_prefix_and_metadata():
    chunks = chunk_markdown(SMALL, source="mod/doc.md", chunk_size=500)

    c = chunks[0]
    assert c["text"].startswith("mod/doc > Doc")
    assert c["source"] == "mod/doc.md"
    assert c["index"] == 0


def test_header_path_has_no_markdown_emphasis():
    chunks = chunk_markdown("# Doc\n\n## **What to expect**\n\nbody", source="mod/doc.md")

    assert chunks[0]["headers"] == ["Doc", "What to expect"]
    assert "**" not in chunks[0]["text"].splitlines()[0]


def test_headerless_text_still_chunks():
    chunks = chunk_markdown("just plain text", source="mod/plain.md", chunk_size=500)

    assert len(chunks) == 1
    assert "just plain text" in chunks[0]["text"]
    assert chunks[0]["headers"] == []


def test_empty_text_gives_no_chunks():
    assert chunk_markdown("  \n", source="mod/empty.md") == []


def test_oversized_sections_are_cut_between_sentences_not_inside_them():
    paragraph = " ".join(f"Sentence number {i} is complete." for i in range(60))

    chunks = chunk_markdown("# Doc\n\n" + paragraph, source="mod/doc.md", chunk_size=300, overlap=60)

    bodies = [c["text"].split("\n", 1)[1].strip() for c in chunks]
    assert len(bodies) > 3
    assert all(body.endswith(".") for body in bodies)
    assert all(body.startswith(("Sentence", "# Doc")) for body in bodies)


def test_a_header_is_never_left_alone_in_its_own_chunk():
    paragraph = " ".join(f"Sentence number {i} is complete." for i in range(60))

    chunks = chunk_markdown("# Doc\n\n" + paragraph, source="mod/doc.md", chunk_size=300, overlap=60)

    assert chunks[0]["text"].split("\n", 1)[1].startswith("# Doc\nSentence number 0")
    assert all(len(c["text"].split("\n", 1)[1]) > 40 for c in chunks)
    assert all(len(c["text"].split("\n", 1)[1]) <= 300 for c in chunks)
