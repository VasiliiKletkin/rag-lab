"""Скачивает учебный корпус в data/ (только стандартная библиотека, можно запускать повторно).

Запуск: uv run python scripts/fetch_data.py
"""

import html
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = [
    {
        "file": "python-3.14-tutorial-datastructures.rst",
        "url": "https://raw.githubusercontent.com/python/cpython/3.14/Doc/tutorial/datastructures.rst",
        "title": "Python 3.14 Tutorial: Data Structures",
        "license": "PSF License Version 2 (Python documentation)",
    },
    {
        "file": "python-3.14-asyncio-task.rst",
        "url": "https://raw.githubusercontent.com/python/cpython/3.14/Doc/library/asyncio-task.rst",
        "title": "Python 3.14 Library Reference: Coroutines and Tasks (asyncio)",
        "license": "PSF License Version 2 (Python documentation)",
    },
    {
        "file": "postgresql-16-index-types.txt",
        "url": "https://www.postgresql.org/docs/16/indexes-types.html",
        "title": "PostgreSQL 16 Documentation: 11.2. Index Types",
        "license": "PostgreSQL License",
    },
    {
        "file": "postgresql-16-textsearch-intro.txt",
        "url": "https://www.postgresql.org/docs/16/textsearch-intro.html",
        "title": "PostgreSQL 16 Documentation: 12.1. Introduction (Full Text Search)",
        "license": "PostgreSQL License",
    },
    {
        "file": "pgvector-readme.md",
        "url": "https://raw.githubusercontent.com/pgvector/pgvector/v0.8.7/README.md",
        "title": "pgvector 0.8.7 README",
        "license": "PostgreSQL License (pgvector)",
    },
]

BLOCK_TAGS = {"p", "div", "pre", "li", "dt", "dd", "tr", "table", "h1", "h2", "h3", "h4", "br"}
SKIP_CLASSES = {"navheader", "navfooter", "toc"}


class DocContentParser(HTMLParser):
    """Собирает текст внутри <div id="docContent"> страницы документации postgresql.org."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0  # глубина вложенности div внутри docContent, 0 = снаружи
        self.skip_at = 0  # глубина пропускаемого div навигации/оглавления, 0 = нет
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.depth == 0:
            if tag == "div" and ("id", "docContent") in attrs:
                self.depth = 1
            return
        if tag == "div":
            self.depth += 1
            if not self.skip_at and dict(attrs).get("class") in SKIP_CLASSES:
                self.skip_at = self.depth
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.depth == 0:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "div":
            if self.skip_at == self.depth:
                self.skip_at = 0
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth > 0 and not self.skip_at:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts))
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "rag-lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for source in SOURCES:
        body = fetch(source["url"])
        if source["url"].endswith(".html"):
            parser = DocContentParser()
            parser.feed(body)
            body = parser.text()
        (DATA_DIR / source["file"]).write_text(body, encoding="utf-8")
        print(f"{source['file']}: {len(body):,} chars")
    manifest = json.dumps(SOURCES, indent=2, ensure_ascii=False) + "\n"
    (DATA_DIR / "manifest.json").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
