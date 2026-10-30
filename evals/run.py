"""Прогон метрик поиска по evals/questions.jsonl на живой базе и моделях.

    uv run python -m evals.run                 # все режимы, k=5
    uv run python -m evals.run --modes vector  # один режим
    uv run python -m evals.run --write         # дописать таблицу в evals/results.md

LLM-as-judge (качество ответа) здесь нет: он требует вызовов модели.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.db import connect
from app.rag.embeddings import get_embedder
from app.rag.retrieval import SearchMode, search
from evals.metrics import Question, gold_rank, mrr, recall_at_k

QUESTIONS = Path(__file__).with_name("questions.jsonl")
RESULTS = Path(__file__).with_name("results.md")


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [
        Question(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def evaluate(mode: SearchMode, questions: list[Question], k: int) -> dict:
    answerable = [q for q in questions if q.must_contain is not None]
    ranks: list[int | None] = []
    misses: list[str] = []
    latencies: list[float] = []
    embedder = get_embedder()
    with connect() as conn:
        for q in answerable:
            t0 = time.perf_counter()
            hits = search(conn, embedder, q.question, k, mode)
            latencies.append(time.perf_counter() - t0)
            rank = gold_rank(q, [(h.document, h.text) for h in hits])
            ranks.append(rank)
            if rank is None:
                misses.append(q.id)
    latencies.sort()
    return {
        "mode": mode,
        "n": len(answerable),
        f"recall@{k}": recall_at_k(ranks, k),
        "mrr": mrr(ranks),
        "p50_ms": latencies[len(latencies) // 2] * 1000,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid", "rerank"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--write", action="store_true", help="дописать результаты в results.md")
    args = parser.parse_args()

    settings = get_settings()
    questions = load_questions()
    rows = [evaluate(mode, questions, args.k) for mode in args.modes]

    header = (
        f"| режим | n | recall@{args.k} | MRR | p50, мс | промахи |\n|---|---|---|---|---|---|\n"
    )
    table = header + "".join(
        f"| {r['mode']} | {r['n']} | {r[f'recall@{args.k}']:.2f} | {r['mrr']:.2f} "
        f"| {r['p50_ms']:.0f} | {', '.join(r['misses']) or '—'} |\n"
        for r in rows
    )
    config = (
        f"chunker fixed-{settings.chunk_size}-{settings.chunk_overlap}, "
        f"embedder {settings.embedding_model}, reranker {settings.reranker_model}, "
        f"candidates {settings.candidates}"
    )
    print(config)
    print(table)
    if args.write:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(f"\n## {stamp}\n\n{config}\n\n{table}")


if __name__ == "__main__":
    main()
