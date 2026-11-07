"""Score hybrid retrieval (vector + keyword, fused with RRF) on golden sets.

    uv run python -m rag.evaluation.report
"""

import argparse
from pathlib import Path

from rag.embed import MODELS, load_embedder
from rag.evaluation.golden import Question, load_golden
from rag.evaluation.metrics import evaluate
from rag.retrieval import HybridRetriever
from rag.store import connect

DEFAULT_DB = "postgresql://rag:rag@127.0.0.1:5433/rag"
DEFAULT_SETS = [
    Path("eval/golden_set.json"),
    Path("eval/golden_terms.json"),
    Path("eval/golden_validation.json"),
]


def run(conn, embedder, golden_sets: dict[str, list[Question]], k: int = 5) -> list[dict]:
    retriever = HybridRetriever(conn, embedder)
    return [
        {"set": set_name, "mode": "hybrid", **evaluate(retriever.search, questions, k)}
        for set_name, questions in golden_sets.items()
    ]


def to_markdown(rows: list[dict], k: int) -> str:
    lines = [f"| set | mode | n | hit@{k} | MRR |", "|---|---|---|---|---|"]
    lines += [f"| {r['set']} | {r['mode']} | {r['n']} | {r['hit_rate']:.3f} | {r['mrr']:.3f} |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", choices=sorted(MODELS), default="bge-m3")
    parser.add_argument("--golden", type=Path, nargs="+", default=DEFAULT_SETS)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    sets = {path.stem: load_golden(path) for path in args.golden}
    with connect(args.db) as conn:
        print(to_markdown(run(conn, load_embedder(args.model), sets, args.k), args.k))


if __name__ == "__main__":
    main()
