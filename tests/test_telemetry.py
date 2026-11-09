import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetry
from app.rag.generation import ClaudeGenerator
from app.rag.retrieval import search
from tests.test_generation import HITS, generator, message
from tests.test_retrieval import FakeReranker, LengthEmbedder, corpus  # noqa: F401


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    # глобальный провайдер уже задан при импорте app.main — подменяем tracer модуля напрямую
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    saved = telemetry.tracer
    telemetry.tracer = tracer
    import app.rag.generation as g
    import app.rag.retrieval as r

    g.tracer = r.tracer = tracer
    yield exporter
    g.tracer = r.tracer = telemetry.tracer = saved


def test_rerank_search_produces_retrieve_and_rerank_spans(corpus, spans) -> None:  # noqa: F811
    search(
        corpus, LengthEmbedder(), "HNSW", k=2, mode="rerank", candidates=10, reranker=FakeReranker()
    )

    names = [s.name for s in spans.get_finished_spans()]
    assert names == ["embed_query", "retrieve", "rerank"]  # вложенный спан завершается раньше
    retrieve = next(s for s in spans.get_finished_spans() if s.name == "retrieve")
    assert retrieve.attributes["rag.mode"] == "rerank"
    assert retrieve.attributes["rag.store"] == "pgvector"


def test_generate_span_records_tokens(spans) -> None:
    gen, _ = generator(message())
    assert isinstance(gen, ClaudeGenerator)

    gen.answer("x", HITS)

    span = spans.get_finished_spans()[0]
    assert span.name == "generate"
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
