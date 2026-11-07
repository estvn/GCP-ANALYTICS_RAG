"""Golden set: questions plus the passages that answer them."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Evidence:
    source: str  # path of the markdown file, relative to the corpus root
    text: str  # snippet copied from that file; chunk-size independent


@dataclass(frozen=True)
class Question:
    id: str
    lang: str  # language of the question, "es" or "en"
    text: str
    relevant: tuple[Evidence, ...]  # any one of them answers the question


def load_golden(path, source_prefix: str = "") -> list[Question]:
    """Load a golden set. `source_prefix` is prepended to every source (for an index built from a parent folder)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["questions"]
    questions = [
        Question(
            id=item["id"],
            lang=item["lang"],
            text=item["question"],
            relevant=tuple(
                Evidence(source_prefix + e["source"], e["evidence"])
                for e in item["relevant"] + item.get("also_relevant", [])
            ),
        )
        for item in raw
    ]
    ids = [question.id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids in golden set")
    return questions
