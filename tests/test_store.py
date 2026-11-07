import numpy as np

from rag.store import replace_document, vector_search


def test_the_vector_dimension_is_configurable(make_db):
    conn = make_db(dim=8)
    chunks = [
        {"index": 0, "headers": [], "text": "alpha"},
        {"index": 1, "headers": [], "text": "beta"},
    ]
    vectors = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    replace_document(conn, "a.md", "hash", chunks, vectors)

    hits = vector_search(conn, np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), 2)

    assert [h["text"] for h in hits] == ["beta", "alpha"]
