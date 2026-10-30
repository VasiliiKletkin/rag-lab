from fastapi.testclient import TestClient

import app.main as main
from app.rag.generation import GenerationError
from app.schemas import AskResponse, Citation, SearchHit


def test_ask_returns_answer_with_citations(monkeypatch) -> None:
    hits = [SearchHit(chunk_id="1", document="pg.md", text="HNSW", score=1)]

    class Gen:
        def answer(self, question, got_hits):
            assert got_hits == hits
            return AskResponse(
                answer="HNSW", citations=[Citation(document="pg.md", chunk_id="1", quote="HNSW")]
            )

    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "run_search", lambda *a, **kw: hits)
    monkeypatch.setattr(main, "get_generator", lambda: Gen())

    response = TestClient(main.app).post("/ask", json={"question": "Какие индексы?"})

    assert response.status_code == 200
    assert response.json()["citations"][0]["chunk_id"] == "1"


def test_ask_without_key_is_502(monkeypatch) -> None:
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "run_search", lambda *a, **kw: [])

    def no_key():
        raise GenerationError("ANTHROPIC_API_KEY не задан", retryable=False)

    monkeypatch.setattr(main, "get_generator", no_key)

    response = TestClient(main.app).post("/ask", json={"question": "x"})

    assert response.status_code == 502
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
