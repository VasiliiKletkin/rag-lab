from contextlib import contextmanager

from anthropic.types import TextBlock
from fastapi.testclient import TestClient

import app.main as main
from app.rag.generation import NO_ANSWER, ClaudeGenerator
from app.schemas import AskResponse
from tests.test_ask import FakeConn
from tests.test_generation import HITS, cited, message


class FakeStream:
    def __init__(self, chunks: list[str], final) -> None:
        self.text_stream = iter(chunks)
        self._final = final

    def get_final_message(self):
        return self._final


class StreamingClient:
    def __init__(self, chunks: list[str], final) -> None:
        self.messages = self
        self._stream = FakeStream(chunks, final)
        self.calls: list[dict] = []

    @contextmanager
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield self._stream


def test_stream_yields_text_then_response_with_citations() -> None:
    final = message(cited("pgvector поддерживает HNSW и IVFFlat.", 0, "HNSW and IVFFlat"))
    client = StreamingClient(["pgvector ", "поддерживает ", "HNSW и IVFFlat."], final)
    gen = ClaudeGenerator(client, "claude-sonnet-5-5")  # type: ignore[arg-type]

    parts = list(gen.stream("Какие индексы?", HITS))

    assert parts[:-1] == ["pgvector ", "поддерживает ", "HNSW и IVFFlat."]
    assert isinstance(parts[-1], AskResponse)
    assert parts[-1].citations[0].quote == "HNSW and IVFFlat"
    assert client.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_stream_without_hits_is_no_answer_without_api_call() -> None:
    client = StreamingClient([], message())
    gen = ClaudeGenerator(client, "claude-sonnet-5-5")  # type: ignore[arg-type]

    assert list(gen.stream("x", [])) == [AskResponse(answer=NO_ANSWER, citations=[])]
    assert client.calls == []


def test_sse_endpoint_emits_tokens_and_done(monkeypatch) -> None:
    final = message(TextBlock(type="text", text="HNSW", citations=None))
    client = StreamingClient(["HN", "SW"], final)
    gen = ClaudeGenerator(client, "claude-sonnet-5-5")  # type: ignore[arg-type]
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "run_search", lambda *a, **kw: HITS)
    monkeypatch.setattr(main, "get_generator", lambda: gen)

    with TestClient(main.app).stream("POST", "/ask/stream", json={"question": "?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = [e for e in body.split("\n\n") if e]
    assert events[0] == 'event: token\ndata: "HN"'
    assert events[1] == 'event: token\ndata: "SW"'
    assert events[2].startswith("event: done\ndata: ") and '"answer":"HNSW"' in events[2]


def test_ask_uses_answer_cache(monkeypatch) -> None:
    calls = []

    class Gen:
        def answer(self, question, hits):
            calls.append(question)
            return AskResponse(answer="HNSW", citations=[])

    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "run_search", lambda *a, **kw: HITS)
    monkeypatch.setattr(main, "get_generator", lambda: Gen())
    monkeypatch.setattr(main, "index_version", lambda conn: "v1")
    main.answer_cache.clear()
    api = TestClient(main.app)

    api.post("/ask", json={"question": "Какие индексы?"})
    api.post("/ask", json={"question": "какие  индексы"})  # то же после нормализации
    monkeypatch.setattr(main, "index_version", lambda conn: "v2")  # корпус изменился
    api.post("/ask", json={"question": "Какие индексы?"})

    assert len(calls) == 2
    assert api.get("/cache/stats").json() == {"hits": 1, "misses": 2, "size": 2, "hit_rate": 0.333}
