import numpy as np

from rag.evaluation import load_golden
from rag.evaluation.report import load_translations, run, to_markdown
from rag.ingestion.pipeline import ingest
from rag.store import connect

MODES = ["vector", "vector + ES/EN 1:1", "vector + ES/EN 2:1"]


class WordEmbedder:
    """Puts a text on one axis chosen by its first letter, so retrieval is predictable."""

    def count_tokens(self, text):
        return len(text.split())

    def embed(self, texts, kind):
        rows = []
        for t in texts:
            v = np.full(384, 0.01, dtype=np.float32)
            v[ord(t.strip().lower()[0]) % 300] += 1.0
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)


def golden_set(tmp_path, *questions):
    """Two documents (a.md holds the answer) and a golden file with the given (id, text) questions."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Z\n\nThe zorblax protocol is rare.", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# B\n\nSomething else entirely.", encoding="utf-8")
    items = ",".join(
        f'{{"id":"{qid}","lang":"es","question":"{text}","relevant":[{{"source":"a.md","evidence":"zorblax protocol is rare"}}]}}'
        for qid, text in questions
    )
    path = tmp_path / "g.json"
    path.write_text('{"questions":[' + items + "]}", encoding="utf-8")
    return tmp_path / "docs", load_golden(path)


def test_run_scores_every_mode_on_every_set(conn, tmp_path):
    docs, questions = golden_set(tmp_path, ("q1", "aaaa"))
    ingest(docs, conn, WordEmbedder())

    rows = run(conn, WordEmbedder(), {"terms": questions}, {"q1": "aaaa"}, k=5)

    assert [(r["set"], r["mode"]) for r in rows] == [("terms", mode) for mode in MODES]


def test_the_translation_rescues_a_question_the_original_phrasing_misses(conn, tmp_path):
    docs, questions = golden_set(tmp_path, ("q1", "??"))  # no word: the original phrasing finds nothing
    ingest(docs, conn, WordEmbedder())

    rows = run(conn, WordEmbedder(), {"terms": questions}, {"q1": "aaaa"}, k=1)

    hit_rates = {r["mode"]: r["hit_rate"] for r in rows}
    assert hit_rates["vector"] == 0.0
    assert hit_rates["vector + ES/EN 1:1"] == hit_rates["vector + ES/EN 2:1"] == 1.0


def test_a_question_without_translation_runs_with_its_single_phrasing(conn, tmp_path):
    docs, questions = golden_set(tmp_path, ("q1", "aaaa"))
    ingest(docs, conn, WordEmbedder())

    rows = run(conn, WordEmbedder(), {"terms": questions}, {}, k=1)

    assert {r["hit_rate"] for r in rows} == {1.0}


def test_translations_are_read_from_the_json_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('{"translations": {"q1": "hello"}}', encoding="utf-8")

    assert load_translations(path) == {"q1": "hello"}


def test_markdown_table_has_one_line_per_row():
    rows = [{"set": "s", "mode": "vector", "n": 10, "hit_rate": 0.7, "mrr": 0.5}]

    table = to_markdown(rows, k=5).splitlines()

    assert table[0].startswith("| set |") and "hit@5" in table[0]
    assert table[-1] == "| s | vector | 10 | 0.700 | 0.500 |"


def test_connect_returns_a_connection_that_understands_vectors(conn):
    url = f"postgresql://rag:rag@127.0.0.1:5433/{conn.info.dbname}"

    with connect(url) as c:
        assert c.execute("SELECT '[1,2]'::halfvec(2)").fetchone() is not None
