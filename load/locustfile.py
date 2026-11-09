"""Этап 10: нагрузочный тест. Запуск:

    uv run locust -f load/locustfile.py --host http://localhost:8000 --headless \\
        -u 10 -r 2 -t 60s --csv docs/load/search

Без ключа API нагружаем только /search (rerank — самый тяжёлый режим: эмбеддинг + cross-encoder).
/ask включается через ASK=1, когда ключ есть: тогда видно и эффект кэша ответов.
"""

import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "What index types does pgvector support?",
    "How does HNSW differ from IVFFlat?",
    "How to use a list as a stack in Python?",
    "What does list.sort do with the key argument?",
    "How are query planner statistics collected?",
    "Какие индексы поддерживает pgvector?",
    "Как работает ef_search в HNSW?",
]


class Reader(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def search_rerank(self) -> None:
        self.client.post(
            "/search", json={"query": random.choice(QUESTIONS), "k": 5, "mode": "rerank"}
        )

    @task(1)
    def search_vector(self) -> None:
        self.client.post(
            "/search", json={"query": random.choice(QUESTIONS), "k": 5, "mode": "vector"}
        )

    if os.environ.get("ASK"):

        @task(1)
        def ask(self) -> None:
            # повторяющиеся вопросы → часть запросов попадёт в кэш ответов
            self.client.post("/ask", json={"question": random.choice(QUESTIONS[:3]), "k": 5})
