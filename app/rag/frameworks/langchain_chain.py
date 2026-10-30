"""LangChain: наш поиск как Retriever + цепочка на LCEL (оператор |)."""

from functools import lru_cache
from typing import Any

import psycopg
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.config import get_settings
from app.rag.embeddings import Embedder
from app.rag.frameworks.common import SYSTEM, format_context, response_from
from app.rag.retrieval import search
from app.schemas import AskResponse, SearchHit


class RagLabRetriever(BaseRetriever):
    """Обёртка над app.rag.retrieval.search в интерфейсе LangChain."""

    conn: Any
    embedder: Any
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        hits = search(self.conn, self.embedder, query, self.k, "rerank")
        return [
            Document(
                page_content=h.text,
                metadata={"document": h.document, "chunk_id": h.chunk_id, "score": h.score},
            )
            for h in hits
        ]


PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Документы:\n\n{context}\n\nВопрос: {question}")]
)


def to_hits(docs: list[Document]) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=d.metadata["chunk_id"],
            document=d.metadata["document"],
            text=d.page_content,
            score=d.metadata["score"],
        )
        for d in docs
    ]


def build_chain(retriever: BaseRetriever, llm: BaseChatModel):
    """{"context", "question"} -> prompt -> llm -> str. Это и есть LCEL."""
    return (
        {
            "context": retriever | RunnableLambda(lambda docs: format_context(to_hits(docs))),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(
    conn: psycopg.Connection, embedder: Embedder, llm: BaseChatModel, question: str, k: int
) -> AskResponse:
    retriever = RagLabRetriever(conn=conn, embedder=embedder, k=k)
    hits = to_hits(
        retriever.invoke(question)
    )  # второй вызов поиска — цена абстракции, см. docs/comparison.md
    answer = build_chain(retriever, llm).invoke(question)
    return response_from(answer, hits)


@lru_cache
def get_llm() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    from app.rag.generation import GenerationError

    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not key:
        raise GenerationError("ANTHROPIC_API_KEY не задан", retryable=False)
    return ChatAnthropic(model=settings.llm_model, api_key=key, max_tokens=2048)
