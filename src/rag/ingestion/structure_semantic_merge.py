"""Structure first, then merge by meaning: neighbouring pieces that talk about the same thing become one chunk."""

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.ingestion.semantic import SemanticChunker

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class StructureSemanticMergeChunker:
    """Needs an embedder with `embed(texts, kind)` and `count_tokens(text)`.

    Stage 1: split by headers; a section above `max_tokens` is subdivided by the semantic (cut) chunker.
    Stage 2: embed every piece and repeatedly merge the most similar pair of neighbours in the document. A pair
    merges when its similarity reaches the threshold (a percentile of the document's own neighbour similarities) or
    when one side is below `min_tokens`, and never past `max_tokens`. Merging may cross a section boundary; the merged
    chunk keeps the headers of its first piece.
    """

    def __init__(self, embedder, max_tokens: int = 440, min_tokens: int = 100, percentile: float = 25):
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.percentile = percentile
        self.semantic = SemanticChunker(embedder, max_tokens, min_tokens, 90)

    @property
    def fingerprint(self) -> str:
        return f"structure-semantic-merge:{self.max_tokens}:{self.min_tokens}:{self.percentile}"

    def chunk(self, text: str, source: str) -> list[dict]:
        pieces = self._pieces(text, source)
        if not pieces:
            return []
        vectors = self.embedder.embed([body for _, body in pieces], kind="passage")
        neighbours = np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(neighbours, self.percentile)) if len(neighbours) else -1.0

        groups = [[i] for i in range(len(pieces))]  # indexes of the pieces in each group, in reading order
        sums = [v.astype(np.float32) for v in vectors]
        tokens = [self.embedder.count_tokens(body) for _, body in pieces]

        while len(groups) > 1:
            best, best_similarity = None, -2.0
            for g in range(len(groups) - 1):
                if tokens[g] + tokens[g + 1] > self.max_tokens:
                    continue
                similarity = float(sums[g] @ sums[g + 1] / (np.linalg.norm(sums[g]) * np.linalg.norm(sums[g + 1])))
                small = min(tokens[g], tokens[g + 1]) < self.min_tokens
                if (similarity >= threshold or small) and similarity > best_similarity:
                    best, best_similarity = g, similarity
            if best is None:
                break
            groups[best : best + 2] = [groups[best] + groups[best + 1]]
            sums[best : best + 2] = [sums[best] + sums[best + 1]]
            tokens[best : best + 2] = [tokens[best] + tokens[best + 1]]

        label = source.removesuffix(".md")
        out = []
        for i, group in enumerate(groups):
            headers = pieces[group[0]][0]
            body = "\n\n".join(pieces[j][1] for j in group)
            out.append(
                {"text": f"{' > '.join([label, *headers])}\n{body}", "source": source, "headers": headers, "index": i}
            )
        return out

    def _pieces(self, text: str, source: str) -> list[tuple[list[str], str]]:
        """Stage 1: header sections, with the big ones cut by meaning. Each piece is (headers, body)."""
        pieces: list[tuple[list[str], str]] = []
        for section in MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text):
            body = section.page_content.strip()
            if not body:
                continue
            headers = [section.metadata[k].replace("*", "").strip() for _, k in HEADERS if k in section.metadata]
            if self.embedder.count_tokens(body) <= self.max_tokens:
                pieces.append((headers, body))
                continue
            for chunk in self.semantic.chunk(body, source):
                pieces.append((chunk["headers"], chunk["text"].split("\n", 1)[1]))
        return pieces
