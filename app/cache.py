"""Этап 10: кэш ответов.

Ключ — нормализованный вопрос + k + версия индекса. Версия индекса меняется при каждом
изменении корпуса, поэтому после /ingest старые ответы сами становятся недостижимыми:
инвалидировать по событию не нужно. Кэш в памяти процесса — для нескольких воркеров
его место занял бы Redis с тем же ключом.
"""

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import psycopg

# версия индекса = сколько кусков с векторами и когда последний раз менялся документ
INDEX_VERSION_SQL = """
SELECT count(c.id)::text || ':' || coalesce(max(d.ingested_at)::text, '') || ':'
       || coalesce(max(c.embedding_model), '')
FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.embedding IS NOT NULL
"""


def index_version(conn: psycopg.Connection) -> str:
    return conn.execute(INDEX_VERSION_SQL).fetchone()[0]


def normalize(question: str) -> str:
    """«Какие индексы в pgvector?» и «какие  индексы в pgvector» — один вопрос."""
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?!. ")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class LRUCache[T]:
    """LRU с TTL. OrderedDict: move_to_end при чтении, popitem(last=False) выталкивает старейший."""

    def __init__(self, maxsize: int = 512, ttl_seconds: float = 3600) -> None:
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self.stats = CacheStats()

    def get(self, key: str) -> T | None:
        item = self._data.get(key)
        if item is None or item[0] < time.monotonic():
            if item is not None:
                del self._data[key]
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return item[1]

    def put(self, key: str, value: T) -> None:
        self._data[key] = (time.monotonic() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)
        self.stats.size = len(self._data)

    def clear(self) -> None:
        self._data.clear()
        self.stats = CacheStats()


def answer_key(question: str, k: int, version: str) -> str:
    return f"{normalize(question)}|{k}|{version}"
