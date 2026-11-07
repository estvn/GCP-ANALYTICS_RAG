"""Incremental ingestion: markdown folder -> chunks -> embeddings -> Postgres."""

import hashlib
from pathlib import Path

from rag.ingestion.semantic import SemanticChunker
from rag.store import delete_documents, get_hashes, replace_document


def ingest(md_root, conn, embedder, chunker=None) -> dict[str, int]:
    """Sync md_root into the database. Only new or changed files are chunked and embedded.

    A chunker has `chunk(text, source)` and a `fingerprint` that changes with its configuration.
    The default is the semantic chunker with a cut at every header (the winner of E29), driven by `embedder`.
    """
    chunker = chunker or SemanticChunker(embedder)
    root = Path(md_root)
    stats = {"added": 0, "updated": 0, "removed": 0, "skipped": 0}
    known = get_hashes(conn)
    seen = set()

    for f in sorted(root.rglob("*.md")):
        source = f.relative_to(root).as_posix()
        seen.add(source)
        text = f.read_text(encoding="utf-8")
        # the chunker configuration is part of the hash: changing it must re-index
        digest = hashlib.sha256(f"{chunker.fingerprint}:{text}".encode()).hexdigest()
        if known.get(source) == digest:
            stats["skipped"] += 1
            continue
        chunks = chunker.chunk(text, source)
        vectors = embedder.embed([c["text"] for c in chunks], kind="passage") if chunks else []
        replace_document(conn, source, digest, chunks, vectors)
        stats["updated" if source in known else "added"] += 1

    gone = set(known) - seen
    delete_documents(conn, gone)
    stats["removed"] = len(gone)
    return stats
