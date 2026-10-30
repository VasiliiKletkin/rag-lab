from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    written: int = Field(description="Новые и изменённые документы")
    unchanged: int = Field(description="Документы без изменений")
    deleted: int = Field(description="Документы, которых больше нет в manifest")
    chunks: int = Field(description="Сколько кусков записано за этот запуск")
    embedded: int = Field(description="Скольким кускам посчитаны векторы")


class SearchRequest(BaseModel):
    query: str = Field(description="Поисковый запрос")
    k: int = Field(5, ge=1, le=50, description="Сколько кусков вернуть")
    mode: Literal["vector", "hybrid", "rerank"] = Field("vector", description="Режим поиска")


class SearchHit(BaseModel):
    chunk_id: str
    document: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(description="Вопрос по корпусу документов")
    k: int = Field(5, ge=1, le=20, description="Сколько кусков дать модели как контекст")
    engine: Literal["plain", "langchain", "llamaindex"] = Field(
        "plain",
        description="plain — свой код и Claude API; langchain / llamaindex — то же на фреймворке",
    )


class Citation(BaseModel):
    document: str
    chunk_id: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
