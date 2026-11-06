import logging
from dataclasses import asdict

import psycopg
from fastapi import FastAPI, HTTPException, status

from app import __version__
from app.config import get_settings
from app.db import connect, pgvector_version
from app.rag.agent import run_agent
from app.rag.embeddings import get_embedder
from app.rag.frameworks import langchain_chain, llamaindex_engine
from app.rag.generation import GenerationError, get_client, get_generator
from app.rag.ingest import embed_missing, ingest_documents
from app.rag.loader import load_corpus
from app.rag.retrieval import search as run_search
from app.schemas import (
    AgentRequest,
    AgentResponse,
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
    """Найти top-k кусков (режим rerank) и ответить по ним с цитатами."""
    try:
        with connect() as conn:
            if req.engine == "langchain":
                return langchain_chain.ask(
                    conn, get_embedder(), langchain_chain.get_llm(), req.question, req.k
                )
            if req.engine == "llamaindex":
                return llamaindex_engine.ask(
                    conn, get_embedder(), llamaindex_engine.get_llm(), req.question, req.k
                )
            hits = run_search(conn, get_embedder(), req.question, req.k, "rerank")
        return get_generator().answer(req.question, hits)
    except psycopg.OperationalError as exc:
        logger.warning("поиск не прошёл: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "база недоступна") from exc
    except GenerationError as exc:
        # 502/503 — проблема у модели, не наш 500
        code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.retryable else status.HTTP_502_BAD_GATEWAY
        logger.warning("генерация не прошла: %s", exc)
        raise HTTPException(code, str(exc)) from exc


@app.post("/agent/ask")
def agent_ask(req: AgentRequest) -> AgentResponse:
    """Агент: модель сама вызывает поиск столько раз, сколько нужно; в ответе — трасса вызовов."""
    try:
        client = get_client()
        with connect() as conn:
            return run_agent(
                client,
                get_settings().llm_model,
                conn,
                get_embedder(),
                req.question,
                req.max_steps,
            )
    except psycopg.OperationalError as exc:
        logger.warning("агент: база недоступна: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "база недоступна") from exc
    except GenerationError as exc:
        code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.retryable else status.HTTP_502_BAD_GATEWAY
        logger.warning("агент: генерация не прошла: %s", exc)
        raise HTTPException(code, str(exc)) from exc
