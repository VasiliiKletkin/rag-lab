"""Этап 1: чтение корпуса по manifest.json."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    path: str  # путь относительно папки корпуса, например "pgvector-readme.md"
    title: str
    url: str | None
    text: str
    content_hash: str  # sha256 текста, по нему понимаем, что документ изменился


def load_corpus(data_dir: Path) -> list[SourceDocument]:
    """Прочитать документы, перечисленные в `data_dir/manifest.json`.

    Список берём из manifest, а не из содержимого папки: в `data/` лежат ещё README.md
    и сам manifest.json, они в корпус попадать не должны.
    """
    root = data_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents = []
    for entry in manifest:
        file = (root / entry["file"]).resolve()
        if not file.is_relative_to(root):
            raise ValueError(f"файл вне папки корпуса: {entry['file']}")
        text = file.read_text(encoding="utf-8")
        documents.append(
            SourceDocument(
                path=entry["file"],
                title=entry["title"],
                url=entry.get("url"),
                text=text,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return documents
