import numpy as np

from rag.store import keyword_search, replace_document


def test_replacing_a_document_swaps_its_chunks(make_db):
    conn = make_db(dim=8)
    vector = np.zeros((1, 8), dtype=np.float32)
    replace_document(conn, "a.md", "h1", [{"index": 0, "headers": [], "text": "alpha"}], vector)
    replace_document(conn, "a.md", "h2", [{"index": 0, "headers": [], "text": "beta"}], vector)

    assert keyword_search(conn, "alpha", 5) == []
    assert [h["text"] for h in keyword_search(conn, "beta", 5)] == ["beta"]
