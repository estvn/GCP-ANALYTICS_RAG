"""Semantic chunking: cut where the meaning shifts, never past the embedder's token limit."""

import re
from dataclasses import dataclass

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADER = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
BULLET = re.compile(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"“(\d])")
MIN_DISTANCE = 1e-4  # below this, differences between neighbours are numerical noise


@dataclass(frozen=True)
class Unit:
    text: str
    headers: tuple[str, ...]


def split_units(markdown: str) -> list[Unit]:
    """Sentences, bullets and wrapped lines. A header travels with the first unit that follows it."""
    units: list[Unit] = []
    path: list[str] = []
    pending_headers: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer)
        parts = [text] if BULLET.match(buffer[0]) else SENTENCE_END.split(text)
        for part in parts:
            body = "\n".join([*pending_headers, part])
            pending_headers.clear()
            units.append(Unit(body, tuple(path)))
        buffer.clear()

    for line in markdown.splitlines():
        header = HEADER.match(line)
        if header:
            flush()
            level = len(header.group(1))
            path[:] = [*path[: level - 1], header.group(2).replace("*", "").strip()]
            pending_headers.append(line.rstrip())
        elif not line.strip():
            flush()
        elif BULLET.match(line):
            flush()
            buffer.append(line.strip())
        else:
            buffer.append(line.strip())
    flush()
    return units


class SemanticChunker:
    """Cut where the meaning shifts, under a hard token cap; headers are context only, never cut points.

    Needs an embedder with `embed(texts, kind)` and `count_tokens(text)`.
    """

    def __init__(self, embedder, max_tokens: int = 440, min_tokens: int = 100, percentile: float = 90):
        # max_tokens leaves room for the context prefix and special tokens under the 512-token model limit
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.percentile = percentile

    @property
    def fingerprint(self) -> str:
        # the trailing "False" is the old header-cut option, kept so indexes built when it was an option are not re-embedded
        return f"semantic:{self.max_tokens}:{self.min_tokens}:{self.percentile}:False"

    def chunk(self, text: str, source: str) -> list[dict]:
        units = split_units(text)
        shifts = self._topic_shifts(units)
        pieces: list[tuple[tuple[str, ...], str]] = []
        current: list[Unit] = []
        tokens = 0

        def flush() -> None:
            nonlocal tokens
            if current:
                pieces.append((current[0].headers, "\n".join(u.text for u in current)))
            current.clear()
            tokens = 0

        for i, unit in enumerate(units):
            size = self.embedder.count_tokens(unit.text)
            if size > self.max_tokens:
                flush()
                pieces += [(unit.headers, part) for part in self._split_long(unit.text)]
                continue
            if current and tokens + size > self.max_tokens:
                flush()
            current.append(unit)
            tokens += size
            if i in shifts and tokens >= self.min_tokens:
                flush()
        flush()

        label = source.removesuffix(".md")
        return [
            {"text": f"{' > '.join([label, *headers])}\n{body}", "source": source, "headers": list(headers), "index": i}
            for i, (headers, body) in enumerate(pieces)
        ]

    def _topic_shifts(self, units: list[Unit]) -> set[int]:
        """Indexes i where the meaning changes between unit i and unit i+1."""
        if len(units) < 3:
            return set()
        windows = [" ".join(u.text for u in units[max(0, i - 1) : i + 2]) for i in range(len(units))]
        vectors = self.embedder.embed(windows, kind="passage")
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = max(np.percentile(distances, self.percentile), MIN_DISTANCE)
        return {i for i, distance in enumerate(distances) if distance > threshold}

    def _split_long(self, text: str) -> list[str]:
        """A single unit above the cap (rare): split it, measuring size in real tokens."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens, chunk_overlap=0, length_function=self.embedder.count_tokens
        )
        return splitter.split_text(text)
