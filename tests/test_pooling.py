import numpy as np

from rag.embed import BGE_M3, E5_SMALL, MODELS, pool


def test_mean_pooling_averages_only_real_tokens_and_normalises():
    hidden = np.array([[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]])  # the third token is padding
    mask = np.array([[1, 1, 0]])

    vectors = pool(hidden, mask, "mean")

    assert np.allclose(vectors, [[2**-0.5, 2**-0.5]])


def test_cls_pooling_takes_the_first_token_and_normalises():
    hidden = np.array([[[3.0, 4.0], [100.0, 0.0]]])
    mask = np.array([[1, 1]])

    assert np.allclose(pool(hidden, mask, "cls"), [[0.6, 0.8]])


def test_model_specs_describe_their_models():
    assert (E5_SMALL.dim, E5_SMALL.pooling, E5_SMALL.query_prefix, E5_SMALL.passage_prefix) == (
        384,
        "mean",
        "query: ",
        "passage: ",
    )
    assert (BGE_M3.dim, BGE_M3.pooling, BGE_M3.query_prefix, BGE_M3.passage_prefix) == (1024, "cls", "", "")
    assert BGE_M3.max_tokens > E5_SMALL.max_tokens


def test_models_are_selectable_by_name():
    assert set(MODELS) == {"e5-small", "bge-m3"}
