from pathlib import Path

import numpy as np
import pytest

from rag.embed import Embedder

MODEL_DIR = Path(__file__).parent.parent / "models" / "multilingual-e5-small"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="model not downloaded")


@pytest.fixture(scope="module")
def emb():
    return Embedder(MODEL_DIR, threads=2)


def test_shape_and_unit_norm(emb):
    v = emb.embed(["hola mundo", "hello world", "x"], kind="passage")

    assert v.shape == (3, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


def test_related_text_scores_higher_than_unrelated_across_languages(emb):
    q = emb.embed(["¿Qué es BigQuery?"], kind="query")[0]
    near, far = emb.embed(
        ["BigQuery is a serverless cloud data warehouse.", "Boil the pasta for ten minutes."],
        kind="passage",
    )

    assert q @ near > q @ far


def test_query_and_passage_prefixes_differ(emb):
    a = emb.embed(["same text"], kind="query")[0]
    b = emb.embed(["same text"], kind="passage")[0]

    assert not np.allclose(a, b)


def test_batching_does_not_change_the_vector(emb):
    texts = ["short", "a much longer sentence about cloud data warehouses and storage " * 5]
    together = emb.embed(texts, kind="passage")
    alone = np.vstack([emb.embed([t], kind="passage") for t in texts])

    assert np.all(np.sum(together * alone, axis=1) > 0.99)


def test_unknown_kind_is_rejected(emb):
    with pytest.raises(ValueError):
        emb.embed(["x"], kind="document")
