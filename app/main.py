import logging
from dataclasses import asdict

import psycopg
from fastapi import FastAPI, HTTPException, status

from app import __version__
from app.config import get_settings
from app.db import connect, pgvector_version
from app.rag.embeddings import get_embedder
from app.rag.ingest import embed_missing, ingest_documents
from app.rag.loader import load_corpus
from app.rag.retrieval import search as run_search
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="rag-lab",
    version=__version__,
    description="Вопросы и ответы по документам: RAG на FastAPI, pgvector и Claude.",
)


def not_implemented(stage: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"Ещё не реализовано, см. план в README: {stage}",
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: процесс жив. Внешние зависимости не проверяются."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: Postgres доступен и pgvector установлен."""
    try:
        version = pgvector_version()
    except Exception as exc:
        logger.warning("проверка готовности не прошла: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "база недоступна") from exc
    if version is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "нет расширения pgvector")
    return {"status": "ready", "pgvector": version}


@app.post("/ingest")
def ingest() -> IngestResponse:
    """Синхронизировать базу с корпусом из настроек (DATA_DIR/manifest.json).

    Путь к папке намеренно не принимается из запроса: клиент API не должен выбирать,
    какие файлы сервера читать.
    """
    settings = get_settings()
    documents = load_corpus(settings.data_dir)
    try:
        with connect() as conn:
            stats = ingest_documents(conn, documents, settings.chunk_size, settings.chunk_overlap)
            embedded = embed_missing(conn, get_embedder())
    except psycopg.OperationalError as exc:
        logger.warning("загрузка не прошла: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "база недоступна") from exc
    return IngestResponse(**asdict(stats), embedded=embedded)


@app.post("/search")
def search(req: SearchRequest) -> SearchResponse:
    try:
        with connect() as conn:
            hits = run_search(conn, get_embedder(), req.query, req.k, req.mode)
    except psycopg.OperationalError as exc:
        logger.warning("поиск не прошёл: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "база недоступна") from exc
    return SearchResponse(hits=hits)


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    # TODO(этап 4): найти top-k кусков, передать их Claude как документы,
    #   вернуть ответ с цитатами из исходных кусков.
    raise not_implemented("этап 4")
