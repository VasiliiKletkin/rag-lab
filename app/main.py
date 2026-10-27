import logging

from fastapi import FastAPI, HTTPException, status

from app import __version__
from app.db import pgvector_version
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
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
def ingest(req: IngestRequest) -> IngestResponse:
    # TODO(этап 1): прочитать файлы из req.path, нарезать на куски (app/rag/chunking.py),
    #   сохранить документы и куски в Postgres; повторная загрузка неизменённых файлов
    #   ничего не меняет.
    # TODO(этап 2): посчитать эмбеддинги кусков (app/rag/embeddings.py) и записать в pgvector.
    raise not_implemented("этапы 1-2")


@app.post("/search")
def search(req: SearchRequest) -> SearchResponse:
    # TODO(этап 3): векторный поиск (косинус) -> гибридный (полнотекстовый + векторный, RRF)
    #   -> переранжирование.
    raise not_implemented("этап 3")


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    # TODO(этап 4): найти top-k кусков, передать их Claude как документы,
    #   вернуть ответ с цитатами из исходных кусков.
    raise not_implemented("этап 4")
