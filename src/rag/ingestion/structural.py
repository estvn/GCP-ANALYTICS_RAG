"""Structure-first chunking: split by headers, merge small sections, split big ones."""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
# Prefer paragraph, line and sentence ends before falling back to a plain space or a character.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, keep_separator="end"
    )
    return splitter.split_text(text)


def chunk_markdown(text: str, source: str, chunk_size: int = 2400, overlap: int = 200) -> list[dict]:
    """Chunk one Markdown document. Sizes are in characters (~4 chars per token)."""
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)

    def split(text: str, size: int) -> list[str]:
        return _recursive_split(text, size, overlap)

    def split_keeping_header(body: str) -> list[str]:
        """Split an oversized section; its header lines stay attached to the first piece."""
        lines = body.split("\n")
        n_header_lines = next((i for i, line in enumerate(lines) if not line.startswith("#")), len(lines))
        header = "\n".join(line.rstrip() for line in lines[:n_header_lines])
        rest = "\n".join(lines[n_header_lines:]).strip()
        if not header or not rest:
            return split(body, chunk_size)
        parts = split(rest, chunk_size - len(header) - 1)
        parts[0] = f"{header}\n{parts[0]}"
        return parts

    pieces: list[tuple[list[str], str]] = []  # (header path, body)
    buf_headers, buf = [], ""

    def flush():
        nonlocal buf_headers, buf
        if buf:
            pieces.append((buf_headers, buf))
        buf_headers, buf = [], ""

    for s in sections:
        body = s.page_content.strip()
        if not body:
            continue
        headers = [s.metadata[k].replace("*", "").strip() for _, k in HEADERS if k in s.metadata]
        if len(body) > chunk_size:
            flush()
            pieces.extend((headers, p) for p in split_keeping_header(body))
        elif buf and len(buf) + len(body) + 2 > chunk_size:
            flush()
            buf_headers, buf = headers, body
        else:
            buf_headers = buf_headers or headers
            buf = f"{buf}\n\n{body}" if buf else body
    flush()

    label = source.removesuffix(".md")
    return [
        {
            "text": f"{' > '.join([label, *h])}\n{body}",
            "source": source,
            "headers": h,
            "index": i,
        }
        for i, (h, body) in enumerate(pieces)
    ]


class StructuralChunker:
    """Headers first, then size: the chunking used by default."""

    def __init__(self, chunk_size: int = 2400, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def fingerprint(self) -> str:
        return f"structural:{self.chunk_size}:{self.overlap}"

    def chunk(self, text: str, source: str) -> list[dict]:
        return chunk_markdown(text, source, self.chunk_size, self.overlap)
