import hashlib
import json
from pathlib import Path

import pytest

from app.rag.loader import load_corpus


def write_corpus(root: Path, files: dict[str, str], manifest: list[dict]) -> None:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_reads_only_files_from_manifest(tmp_path: Path) -> None:
    write_corpus(
        tmp_path,
        {"a.md": "Привет", "README.md": "не часть корпуса"},
        [{"file": "a.md", "title": "A", "url": "https://example.com/a"}],
    )

    docs = load_corpus(tmp_path)

    assert [d.path for d in docs] == ["a.md"]
    assert docs[0].title == "A"
    assert docs[0].url == "https://example.com/a"
    assert docs[0].content_hash == hashlib.sha256("Привет".encode()).hexdigest()


def test_rejects_files_outside_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "data"
    corpus.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    write_corpus(corpus, {}, [{"file": "../secret.txt", "title": "S"}])

    with pytest.raises(ValueError):
        load_corpus(corpus)


def test_real_corpus_loads() -> None:
    docs = load_corpus(Path("data"))

    assert len(docs) == 5
    assert all(d.text for d in docs)
