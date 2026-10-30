"""LlamaIndex: наш поиск как Retriever + RetrieverQueryEngine."""

from functools import lru_cache
from typing import Any

import psycopg
from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.config import get_settings
from app.rag.embeddings import Embedder
from app.rag.frameworks.common import SYSTEM, response_from
from app.rag.retrieval import search
from app.schemas import AskResponse, SearchHit


class RagLabRetriever(BaseRetriever):
    def __init__(self, conn: Any, embedder: Any, k: int) -> None:
        super().__init__()
        self._conn, self._embedder, self._k = conn, embedder, k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        hits = search(self._conn, self._embedder, query_bundle.query_str, self._k, "rerank")
        return [
            NodeWithScore(
                node=TextNode(text=h.text, id_=h.chunk_id, metadata={"document": h.document}),
                score=h.score,
            )
            for h in hits
        ]


# LlamaIndex сам подставляет куски и вопрос в свой шаблон; наш промпт — поверх него
QA_PROMPT = PromptTemplate(SYSTEM + "\n\nДокументы:\n{context_str}\n\nВопрос: {query_str}\nОтвет: ")


def to_hits(nodes: list[NodeWithScore]) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=n.node.node_id,
            document=n.node.metadata["document"],
            text=n.node.get_content(),
            score=n.score or 0.0,
        )
        for n in nodes
    ]


def ask(
    conn: psycopg.Connection, embedder: Embedder, llm: LLM, question: str, k: int
) -> AskResponse:
    retriever = RagLabRetriever(conn, embedder, k)
    synthesizer = get_response_synthesizer(
        llm=llm, text_qa_template=QA_PROMPT, response_mode="compact"
    )
    engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
    response = engine.query(question)
    return response_from(str(response), to_hits(response.source_nodes))


@lru_cache
def get_llm() -> LLM:
    from llama_index.llms.anthropic import Anthropic

    from app.rag.generation import GenerationError

    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not key:
        raise GenerationError("ANTHROPIC_API_KEY не задан", retryable=False)
    return Anthropic(model=settings.llm_model, api_key=key, max_tokens=2048)
