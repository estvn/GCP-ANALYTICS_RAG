from pathlib import Path

import numpy as np
import pytest

from rag.embed import BGE_M3, Embedder

MODEL_DIR = Path(__file__).parent.parent / "models" / "bge-m3"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="bge-m3 not downloaded")


@pytest.fixture(scope="module")
def emb():
    return Embedder(MODEL_DIR, BGE_M3, threads=2)


def test_vectors_have_1024_dimensions_and_unit_norm(emb):
    v = emb.embed(["hola mundo", "hello world"], kind="passage")

    assert v.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


def test_related_text_scores_far_above_unrelated_across_languages(emb):
    q = emb.embed(["¿Qué es BigQuery?"], kind="query")[0]
    near, far = emb.embed(
        ["BigQuery is a serverless cloud data warehouse.", "Boil the pasta for ten minutes."], kind="passage"
    )

    assert q @ near - q @ far > 0.3


def test_queries_and_passages_share_the_same_text_without_prefixes(emb):
    assert np.allclose(emb.embed(["same text"], kind="query"), emb.embed(["same text"], kind="passage"))


def test_it_reads_chunks_longer_than_512_tokens(emb):
    assert emb.count_tokens("palabra " * 800) > 512
    assert emb.embed(["palabra " * 800], kind="passage").shape == (1, 1024)
