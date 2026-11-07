import pytest

from rag.retrieval.fusion import rrf


def ranked(*ids):
    return [{"id": i} for i in ids]


def ids(results):
    return [r["id"] for r in results]


def test_matches_the_guide_example():
    # 'a' is 1st in vector and 3rd in keyword; 'b' is 2nd in vector and 1st in keyword
    assert ids(rrf([ranked("a", "b", "c"), ranked("b", "d", "a")])) == ["b", "a", "d", "c"]


def test_item_in_both_lists_beats_items_in_only_one():
    result = rrf([ranked("x", "shared"), ranked("y", "shared")])

    assert result[0]["id"] == "shared"


def test_fields_from_both_lists_are_merged():
    result = rrf([[{"id": 1, "distance": 0.2}], [{"id": 1, "kw_score": 0.7}]])

    assert result == [{"id": 1, "distance": 0.2, "kw_score": 0.7, "score": result[0]["score"]}]


def test_empty_inputs_give_empty_output():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_single_list_keeps_its_order():
    assert ids(rrf([ranked("a", "b", "c")])) == ["a", "b", "c"]


def test_without_weights_every_list_counts_the_same():
    lists = [ranked("a", "b", "c"), ranked("b", "d", "a")]

    assert rrf(lists) == rrf(lists, weights=[1, 1])


def test_a_heavier_list_ranks_its_first_hit_above_the_other_lists_first_hit():
    lists = [ranked("original"), ranked("translation")]

    assert ids(rrf(lists, weights=[2, 1])) == ["original", "translation"]
    assert ids(rrf(lists, weights=[1, 2])) == ["translation", "original"]


def test_equal_weights_tie_the_first_hits_of_two_lists():
    first, second = rrf([ranked("original"), ranked("translation")], weights=[1, 1])

    assert first["score"] == second["score"]


def test_weight_scales_the_score_of_every_hit_of_its_list():
    result = rrf([ranked("a")], k=60, weights=[3])

    assert result[0]["score"] == 3 / 61


def test_weights_must_match_the_number_of_lists():
    with pytest.raises(ValueError):
        rrf([ranked("a"), ranked("b")], weights=[1])
