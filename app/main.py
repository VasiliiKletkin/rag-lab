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

app = FastAPI(title="rag-lab", version=__version__)


def not_implemented(stage: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"Not implemented yet, see README roadmap: {stage}",
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. No external dependencies."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: Postgres is reachable and pgvector is installed."""
    try:
        version = pgvector_version()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"database: {exc}") from exc
    if version is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "pgvector extension missing")
    return {"status": "ready", "pgvector": version}


@app.post("/ingest")
def ingest(req: IngestRequest) -> IngestResponse:
    # TODO(stage 1): load files from req.path, split into chunks (app/rag/chunking.py),
    #   store documents + chunks in Postgres; re-ingesting unchanged files is a no-op.
    # TODO(stage 2): embed chunks (app/rag/embeddings.py) and write vectors to pgvector.
    raise not_implemented("stages 1-2")


@app.post("/search")
def search(req: SearchRequest) -> SearchResponse:
    # TODO(stage 3): vector search (cosine) -> hybrid (lexical + vector, RRF) -> rerank.
    raise not_implemented("stage 3")


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    # TODO(stage 4): retrieve top-k chunks, call Claude with the chunks as documents,
    #   return the answer with citations to the source chunks.
    raise not_implemented("stage 4")
