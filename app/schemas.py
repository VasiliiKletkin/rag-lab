from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field("data", description="File or directory to ingest")


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class SearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=50)
    mode: Literal["vector", "hybrid", "rerank"] = "vector"


class SearchHit(BaseModel):
    chunk_id: str
    document: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str
    k: int = Field(5, ge=1, le=20)


class Citation(BaseModel):
    document: str
    chunk_id: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
