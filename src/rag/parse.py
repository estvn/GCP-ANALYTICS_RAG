"""Mirror a course folder as Markdown: PDFs are converted, .md files are copied."""

import shutil
from pathlib import Path

import pymupdf4llm

# Legacy engine: the layout engine turns ligatures (fi, fl) into U+FFFD.
pymupdf4llm.use_layout(False)

SKIP_DIRS = {".obsidian"}


def convert_corpus(src: Path, out: Path) -> int:
    """Convert src into out, return how many files were (re)written."""
    src, out = Path(src), Path(out)
    written = 0
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if not f.is_file() or SKIP_DIRS & set(rel.parts) or f.suffix not in {".pdf", ".md"}:
            continue
        dst = (out / rel).with_suffix(".md")
        if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
            continue  # ponytail: mtime check; content hash arrives with ingestion (T4)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".pdf":
            dst.write_text(pymupdf4llm.to_markdown(f), encoding="utf-8")
        else:
            shutil.copyfile(f, dst)
        written += 1
    return written
