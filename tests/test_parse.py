import pymupdf

from rag.parse import convert_corpus


def _make_pdf(path, text):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)


def _make_corpus(root):
    (root / "mod1").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    _make_pdf(root / "mod1" / "a.pdf", "Hello certificate")
    (root / "mod1" / "b.md").write_text("# Title\n\nbody", encoding="utf-8")
    (root / "mod1" / "c.png").write_bytes(b"png")
    (root / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    (root / ".obsidian" / "x.md").write_text("noise", encoding="utf-8")


def test_pdf_and_md_become_md_mirroring_structure(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_corpus(src)

    convert_corpus(src, out)

    assert "certificate" in (out / "mod1" / "a.md").read_text(encoding="utf-8")
    assert (out / "mod1" / "b.md").read_text(encoding="utf-8") == "# Title\n\nbody"


def test_noise_is_skipped_and_source_untouched(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_corpus(src)

    convert_corpus(src, out)

    assert not (out / ".obsidian").exists()
    assert not (out / "mod1" / "c.png").exists()
    assert (src / "mod1" / "a.pdf").exists()


def test_unchanged_files_are_not_reconverted(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_corpus(src)
    first = convert_corpus(src, out)

    second = convert_corpus(src, out)

    assert first == 2
    assert second == 0
