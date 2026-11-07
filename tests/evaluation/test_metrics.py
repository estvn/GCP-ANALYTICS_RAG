import json

import pytest

from rag.evaluation import Evidence, Question, evaluate, is_relevant, load_golden


def q(qid="q1", source="m/a.md", evidence="cloud data warehouse"):
    return Question(qid, "en", "what is it?", (Evidence(source, evidence),))


def hit(source="m/a.md", text="BigQuery is a cloud   data\nwarehouse."):
    return {"source": source, "text": text}


def test_relevant_when_source_matches_and_evidence_is_in_the_text():
    assert is_relevant(hit(), q())


def test_whitespace_and_line_wraps_do_not_matter():
    assert is_relevant(hit(text="a cloud\n data   warehouse"), q())


def test_wrong_source_is_not_relevant():
    assert not is_relevant(hit(source="m/other.md"), q())


def test_right_source_but_evidence_missing_is_not_relevant():
    assert not is_relevant(hit(text="something unrelated"), q())


def test_any_of_several_acceptable_answers_counts():
    question = Question("q1", "en", "?", (Evidence("m/a.md", "nope"), Evidence("m/b.md", "warehouse")))

    assert is_relevant(hit(source="m/b.md"), question)


def scripted(ranks):
    """A fake search that places the relevant hit at the given rank (None = never)."""

    def search(query, k):
        rank = ranks[query]
        hits = [hit(text="noise") for _ in range(k)]
        if rank is not None and rank <= k:
            hits[rank - 1] = hit()
        return hits

    return search


def questions(*ids):
    return [Question(i, "en", i, (Evidence("m/a.md", "cloud data warehouse"),)) for i in ids]


def test_hit_rate_and_mrr():
    report = evaluate(scripted({"q1": 1, "q2": 3, "q3": None}), questions("q1", "q2", "q3"), k=5)

    assert report["n"] == 3
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report["misses"] == ["q3"]


def test_relevant_hit_beyond_k_is_a_miss():
    report = evaluate(scripted({"q1": 6}), questions("q1"), k=5)

    assert report["hit_rate"] == 0 and report["mrr"] == 0


def test_load_golden_merges_also_relevant(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "lang": "es",
                        "question": "¿Qué es?",
                        "relevant": [{"source": "m/a.md", "evidence": "uno"}],
                        "also_relevant": [{"source": "m/b.md", "evidence": "dos"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    [question] = load_golden(path)

    assert question.text == "¿Qué es?" and question.lang == "es"
    assert [e.source for e in question.relevant] == ["m/a.md", "m/b.md"]


def test_load_golden_rejects_duplicate_ids(tmp_path):
    item = {"id": "q1", "lang": "en", "question": "?", "relevant": [{"source": "a", "evidence": "b"}]}
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"questions": [item, item]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_load_golden_can_prefix_every_source(tmp_path):
    item = {
        "id": "q1",
        "lang": "en",
        "question": "?",
        "relevant": [{"source": "m/a.md", "evidence": "x"}],
        "also_relevant": [{"source": "m/b.md", "evidence": "y"}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"questions": [item]}), encoding="utf-8")

    [question] = load_golden(path, source_prefix="course1/")

    assert [e.source for e in question.relevant] == ["course1/m/a.md", "course1/m/b.md"]
