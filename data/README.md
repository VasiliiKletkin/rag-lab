# Учебный корпус

Пять небольших документов под свободными лицензиями для экспериментов. Форматы
специально разные (reStructuredText, Markdown, простой текст), чтобы загрузчику
на этапе 1 было с чем работать.

Документы на английском — это оригинальная документация. Вопросы и ответы в проекте
на русском, поэтому поиск кросс-языковой (см. README, этапы 2–3).

| Файл | Источник | Лицензия |
|---|---|---|
| `python-3.14-tutorial-datastructures.rst` | [CPython `Doc/tutorial/datastructures.rst`](https://github.com/python/cpython/blob/3.14/Doc/tutorial/datastructures.rst) | [PSF License Version 2](https://docs.python.org/3/license.html) |
| `python-3.14-asyncio-task.rst` | [CPython `Doc/library/asyncio-task.rst`](https://github.com/python/cpython/blob/3.14/Doc/library/asyncio-task.rst) | [PSF License Version 2](https://docs.python.org/3/license.html) |
| `postgresql-16-index-types.txt` | [Документация PostgreSQL 16, 11.2 Index Types](https://www.postgresql.org/docs/16/indexes-types.html) | [PostgreSQL License](https://www.postgresql.org/about/licence/) |
| `postgresql-16-textsearch-intro.txt` | [Документация PostgreSQL 16, 12.1 Full Text Search, введение](https://www.postgresql.org/docs/16/textsearch-intro.html) | [PostgreSQL License](https://www.postgresql.org/about/licence/) |
| `pgvector-readme.md` | [README pgvector v0.8.7](https://github.com/pgvector/pgvector/blob/v0.8.7/README.md) | [PostgreSQL License](https://github.com/pgvector/pgvector/blob/v0.8.7/LICENSE) |

`manifest.json` содержит те же источники в машиночитаемом виде: `title` и `url` пригодятся
как метаданные кусков и в цитатах. Страницы PostgreSQL сконвертированы из HTML в простой
текст, остальное сохранено как опубликовано.

Скачать заново:

```bash
uv run python scripts/fetch_data.py
```

Чтобы расширить корпус, добавь запись в `SOURCES` в `scripts/fetch_data.py`. Бери только
источники, лицензия которых разрешает копирование.
