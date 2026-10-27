# Sample corpus

Five small, openly licensed documents to experiment with. They mix formats on purpose
(reStructuredText, Markdown, plain text), so the loader in stage 1 has something to handle.

| File | Source | License |
|---|---|---|
| `python-3.14-tutorial-datastructures.rst` | [CPython `Doc/tutorial/datastructures.rst`](https://github.com/python/cpython/blob/3.14/Doc/tutorial/datastructures.rst) | [PSF License Version 2](https://docs.python.org/3/license.html) |
| `python-3.14-asyncio-task.rst` | [CPython `Doc/library/asyncio-task.rst`](https://github.com/python/cpython/blob/3.14/Doc/library/asyncio-task.rst) | [PSF License Version 2](https://docs.python.org/3/license.html) |
| `postgresql-16-index-types.txt` | [PostgreSQL 16 docs, 11.2 Index Types](https://www.postgresql.org/docs/16/indexes-types.html) | [PostgreSQL License](https://www.postgresql.org/about/licence/) |
| `postgresql-16-textsearch-intro.txt` | [PostgreSQL 16 docs, 12.1 Full Text Search intro](https://www.postgresql.org/docs/16/textsearch-intro.html) | [PostgreSQL License](https://www.postgresql.org/about/licence/) |
| `pgvector-readme.md` | [pgvector v0.8.7 README](https://github.com/pgvector/pgvector/blob/v0.8.7/README.md) | [PostgreSQL License](https://github.com/pgvector/pgvector/blob/v0.8.7/LICENSE) |

`manifest.json` lists the same sources in machine-readable form (use `title` and `url` as
chunk metadata and in citations). The PostgreSQL pages are converted from HTML to plain text;
everything else is stored as published.

Re-download with:

```bash
uv run python scripts/fetch_data.py
```

To grow the corpus, add an entry to `SOURCES` in `scripts/fetch_data.py`. Keep to sources
whose license allows copying.
