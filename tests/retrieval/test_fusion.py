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
