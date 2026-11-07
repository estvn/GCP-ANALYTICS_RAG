"""Two-stage chunking: headers decide the sections, meaning only subdivides the sections that are too big."""

from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.ingestion.semantic import SemanticChunker

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class StructureThenSemanticChunker:
    """Needs an embedder with `embed(texts, kind)` and `count_tokens(text)`.

    Stage 1: split by headers; consecutive sections below `min_tokens` are merged (never past `max_tokens`).
    Stage 2: a section above `max_tokens` is cut by the semantic chunker. Small sections are never embedded here,
    so this is cheaper than the one-pass hybrid, which embeds every window of the document.
    """

    def __init__(self, embedder, max_tokens: int = 440, min_tokens: int = 100, percentile: float = 90):
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.semantic = SemanticChunker(embedder, max_tokens, min_tokens, percentile)

    @property
    def fingerprint(self) -> str:
        return f"structure-then-semantic:{self.max_tokens}:{self.min_tokens}:{self.semantic.percentile}"

    def chunk(self, text: str, source: str) -> list[dict]:
        sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)
        label = source.removesuffix(".md")
        out: list[dict] = []
        buffer_headers: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        def add(headers: list[str], body: str) -> None:
            text = f"{' > '.join([label, *headers])}\n{body}"
            out.append({"text": text, "source": source, "headers": headers, "index": len(out)})

        def flush() -> None:
            nonlocal buffer_headers, buffer, buffer_tokens
            if buffer:
                add(buffer_headers, "\n\n".join(buffer))
            buffer_headers, buffer, buffer_tokens = [], [], 0

        for section in sections:
            body = section.page_content.strip()
            if not body:
                continue
            headers = [section.metadata[k].replace("*", "").strip() for _, k in HEADERS if k in section.metadata]
            size = self.embedder.count_tokens(body)
            if size > self.max_tokens:
                flush()
                for piece in self.semantic.chunk(body, source):
                    out.append({**piece, "index": len(out)})
            elif buffer and (buffer_tokens >= self.min_tokens or buffer_tokens + size > self.max_tokens):
                flush()
                buffer_headers, buffer, buffer_tokens = headers, [body], size
            else:
                buffer_headers = buffer_headers or headers
                buffer.append(body)
                buffer_tokens += size
        flush()
        return out
