"""Score hybrid retrieval on golden sets, alone and with each question's translation fused in.

    uv run python -m rag.evaluation.report
"""

import argparse
import json
from pathlib import Path

from rag.embed import MODELS, load_embedder
from rag.evaluation.golden import Question, load_golden
from rag.evaluation.metrics import evaluate
from rag.retrieval import HybridRetriever
from rag.store import connect

DEFAULT_DB = "postgresql://rag:rag@127.0.0.1:5433/rag"
DEFAULT_TRANSLATIONS = Path("eval/golden_translations.json")
WEIGHTS = [(1, 1), (2, 1)]  # original:translation
DEFAULT_SETS = [
    Path("eval/golden_set.json"),
    Path("eval/golden_terms.json"),
    Path("eval/golden_validation.json"),
]


def load_translations(path: Path) -> dict[str, str]:
    """Question id -> the question in the other language."""
    return json.loads(path.read_text(encoding="utf-8"))["translations"]


def _translated_search(retriever, phrasings: dict[str, list[str]], weights: tuple[int, int]):
    """Search with the question and, when it has one, its translation, fused with the weights original:translation."""

    def search(text: str, limit: int) -> list[dict]:
        queries = phrasings[text]
        return retriever.search_many(queries, k=limit, weights=weights[: len(queries)])

    return search


def run(
    conn, embedder, golden_sets: dict[str, list[Question]], translations: dict[str, str], k: int = 5
) -> list[dict]:
    retriever = HybridRetriever(conn, embedder)
    rows = []
    for set_name, questions in golden_sets.items():
        phrasings = {q.text: [q.text, translations[q.id]] if q.id in translations else [q.text] for q in questions}
        modes = {"hybrid": retriever.search}
        modes |= {f"hybrid + ES/EN {w[0]}:{w[1]}": _translated_search(retriever, phrasings, w) for w in WEIGHTS}
        rows += [{"set": set_name, "mode": mode, **evaluate(search, questions, k)} for mode, search in modes.items()]
    return rows


def to_markdown(rows: list[dict], k: int) -> str:
    lines = [f"| set | mode | n | hit@{k} | MRR |", "|---|---|---|---|---|"]
    lines += [f"| {r['set']} | {r['mode']} | {r['n']} | {r['hit_rate']:.3f} | {r['mrr']:.3f} |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", choices=sorted(MODELS), default="bge-m3")
    parser.add_argument("--golden", type=Path, nargs="+", default=DEFAULT_SETS)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    sets = {path.stem: load_golden(path) for path in args.golden}
    with connect(args.db) as conn:
        rows = run(conn, load_embedder(args.model), sets, load_translations(args.translations), args.k)
        print(to_markdown(rows, args.k))


if __name__ == "__main__":
    main()
