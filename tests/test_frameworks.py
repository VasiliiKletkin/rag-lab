import psycopg
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from llama_index.core.llms import MockLLM

from app.rag.frameworks import langchain_chain, llamaindex_engine
from app.rag.generation import NO_ANSWER
from tests.test_retrieval import LengthEmbedder, corpus  # noqa: F401  (фикстура)


def test_langchain_chain_answers_with_sources(corpus: psycopg.Connection) -> None:  # noqa: F811
    llm = FakeListChatModel(responses=["HNSW и IVFFlat."])

    result = langchain_chain.ask(corpus, LengthEmbedder(), llm, "HNSW index", k=2)

    assert result.answer == "HNSW и IVFFlat."
    assert [c.document for c in result.citations] == ["pg.md", "py.md"]


def test_langchain_no_answer_has_no_citations(corpus: psycopg.Connection) -> None:  # noqa: F811
    llm = FakeListChatModel(responses=[NO_ANSWER])

    result = langchain_chain.ask(corpus, LengthEmbedder(), llm, "борщ", k=2)

    assert result.citations == []


def test_langchain_prompt_contains_context_and_question() -> None:
    from langchain_core.documents import Document

    docs = [
        Document(page_content="HNSW", metadata={"document": "pg.md", "chunk_id": "1", "score": 1.0})
    ]
    rendered = langchain_chain.PROMPT.invoke(
        {
            "context": langchain_chain.format_context(langchain_chain.to_hits(docs)),
            "question": "Какие индексы?",
        }
    )

    text = rendered.to_string()
    assert "[1] pg.md\nHNSW" in text and "Какие индексы?" in text and NO_ANSWER in text


def test_llamaindex_engine_answers_with_sources(corpus: psycopg.Connection) -> None:  # noqa: F811
    llm = MockLLM(max_tokens=16)  # возвращает сам промпт, обрезанный; нам важны источники

    result = llamaindex_engine.ask(corpus, LengthEmbedder(), llm, "HNSW index", k=2)

    assert result.answer
    assert [c.document for c in result.citations] == ["pg.md", "py.md"]
    assert all(c.chunk_id.isdigit() for c in result.citations)
