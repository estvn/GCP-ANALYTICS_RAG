"""Incremental ingestion: markdown folder -> chunks -> embeddings -> Postgres."""

import hashlib
from pathlib import Path

from rag.chunk import chunk_markdown
from rag.store import delete_documents, get_hashes, replace_document


def ingest(md_root, conn, embedder, chunk_size: int = 2400, overlap: int = 200) -> dict[str, int]:
    """Sync md_root into the database. Only new or changed files are chunked and embedded."""
    root = Path(md_root)
    stats = {"added": 0, "updated": 0, "removed": 0, "skipped": 0}
    known = get_hashes(conn)
    seen = set()

    for f in sorted(root.rglob("*.md")):
        source = f.relative_to(root).as_posix()
        seen.add(source)
        text = f.read_text(encoding="utf-8")
        # chunking params are part of the hash: changing them must re-index
        digest = hashlib.sha256(f"{chunk_size}:{overlap}:{text}".encode()).hexdigest()
        if known.get(source) == digest:
            stats["skipped"] += 1
            continue
        chunks = chunk_markdown(text, source, chunk_size, overlap)
        vectors = embedder.embed([c["text"] for c in chunks], kind="passage") if chunks else []
        replace_document(conn, source, digest, chunks, vectors)
        stats["updated" if source in known else "added"] += 1

    gone = set(known) - seen
    delete_documents(conn, gone)
    stats["removed"] = len(gone)
    return stats
