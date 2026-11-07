import numpy as np

from rag.evaluation import load_golden
from rag.evaluation.report import run, to_markdown
from rag.ingestion.pipeline import ingest
from rag.store import connect


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


def test_run_scores_the_keyword_mode_on_every_set(conn, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Z\n\nThe zorblax protocol is rare.", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# B\n\nSomething else entirely.", encoding="utf-8")
    ingest(tmp_path / "docs", conn, WordEmbedder())
    golden = tmp_path / "g.json"
    golden.write_text(
        '{"questions":[{"id":"q1","lang":"en","question":"zorblax","relevant":'
        '[{"source":"a.md","evidence":"zorblax protocol is rare"}]}]}',
        encoding="utf-8",
    )

    rows = run(conn, {"terms": load_golden(golden)}, k=5)

    assert [(r["set"], r["mode"]) for r in rows] == [("terms", "keyword")]
    assert rows[0]["hit_rate"] == 1.0


def test_markdown_table_has_one_line_per_row():
    rows = [{"set": "s", "mode": "keyword", "n": 10, "hit_rate": 0.7, "mrr": 0.5}]

    table = to_markdown(rows, k=5).splitlines()

    assert table[0].startswith("| set |") and "hit@5" in table[0]
    assert table[-1] == "| s | keyword | 10 | 0.700 | 0.500 |"


def test_connect_returns_a_connection_that_understands_vectors(conn):
    url = f"postgresql://rag:rag@127.0.0.1:5433/{conn.info.dbname}"

    with connect(url) as c:
        assert c.execute("SELECT '[1,2]'::halfvec(2)").fetchone() is not None
