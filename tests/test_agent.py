import json

import psycopg
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from fastapi.testclient import TestClient

import app.main as main
from app.rag.agent import run_agent
from app.rag.generation import NO_ANSWER, GenerationError
from app.rag.tools import TOOLS, ToolError, run_tool
from tests.test_retrieval import LengthEmbedder, corpus  # noqa: F401


def message(*blocks, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_1",
        model="claude-sonnet-5-5",
        role="assistant",
        type="message",
        content=list(blocks),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=20),
    )


def tool_use(name: str, tool_input: dict, block_id: str = "toolu_1") -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=block_id, name=name, input=tool_input)


def text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value, citations=None)


class ScriptedClient:
    """Отдаёт заранее заданные ответы по очереди и запоминает, что ему прислали."""

    def __init__(self, *script: Message) -> None:
        self._script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs) -> Message:
        # история — один и тот же список, который агент дополняет: снимаем копию
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._script.pop(0)


def agent(client: ScriptedClient, conn: psycopg.Connection, question: str = "?", **kwargs):
    return run_agent(client, "claude-sonnet-5-5", conn, LengthEmbedder(), question, **kwargs)  # type: ignore[arg-type]


def test_tool_schemas_are_strict() -> None:
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema.get("required", [])) == set(schema["properties"])


def test_agent_searches_then_answers(corpus: psycopg.Connection) -> None:  # noqa: F811
    client = ScriptedClient(
        message(
            tool_use("search_docs", {"query": "HNSW index", "k": 2, "mode": "hybrid"}),
            stop_reason="tool_use",
        ),
        message(text("HNSW и IVFFlat.")),
    )

    result = agent(client, corpus, "Какие индексы?")

    assert result.answer == "HNSW и IVFFlat."
    assert (result.steps, result.stopped) == (2, "answer")
    assert [(c.name, c.is_error) for c in result.trace] == [("search_docs", False)]
    # второй запрос к модели несёт её же tool_use и наш tool_result с тем же id
    history = client.calls[1]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    tool_result = history[2]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])[0]["document"] == "pg.md"


def test_parallel_tool_calls_go_back_in_one_message(corpus: psycopg.Connection) -> None:  # noqa: F811
    client = ScriptedClient(
        message(
            tool_use("list_documents", {}, "toolu_a"),
            tool_use("search_docs", {"query": "stack", "k": 1, "mode": "vector"}, "toolu_b"),
            stop_reason="tool_use",
        ),
        message(text("ok")),
    )

    result = agent(client, corpus)

    results = client.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert len(result.trace) == 2


def test_bad_arguments_return_to_model_as_error(corpus: psycopg.Connection) -> None:  # noqa: F811
    client = ScriptedClient(
        message(
            tool_use("search_docs", {"query": "x", "k": 500, "mode": "rerank"}),
            stop_reason="tool_use",
        ),
        message(tool_use("get_chunk", {"id": 999999}), stop_reason="tool_use"),
        message(tool_use("drop_tables", {}), stop_reason="tool_use"),
        message(text(NO_ANSWER)),
    )

    result = agent(client, corpus)

    assert [c.is_error for c in result.trace] == [True, True, True]
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert result.answer == NO_ANSWER


def test_step_limit_stops_the_loop(corpus: psycopg.Connection) -> None:  # noqa: F811
    loop = message(tool_use("list_documents", {}), stop_reason="tool_use")
    client = ScriptedClient(loop, loop, loop)

    result = agent(client, corpus, max_steps=2)

    assert len(client.calls) == 2
    assert (result.answer, result.stopped) == (NO_ANSWER, "max_steps")


def test_refusal_is_an_error(corpus: psycopg.Connection) -> None:  # noqa: F811
    with pytest.raises(GenerationError):
        agent(ScriptedClient(message(stop_reason="refusal")), corpus)


def test_get_chunk_returns_neighbours(db: psycopg.Connection) -> None:
    from app.rag.ingest import ingest_documents
    from tests.test_ingest import doc

    ingest_documents(db, [doc("a.md", "a" * 10 + "b" * 10 + "c" * 10)], size=10, overlap=0)
    middle = db.execute("SELECT id FROM chunks WHERE chunk_index = 1").fetchone()[0]

    chunks = json.loads(run_tool(db, LengthEmbedder(), "get_chunk", {"id": middle}))

    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]


def test_extra_argument_is_rejected(db: psycopg.Connection) -> None:
    with pytest.raises(ToolError, match="неверные аргументы"):
        run_tool(db, LengthEmbedder(), "get_chunk", {"id": 1, "sql": "DROP TABLE chunks"})


def test_endpoint_without_key_is_502(monkeypatch) -> None:
    def no_key():
        raise GenerationError("ANTHROPIC_API_KEY не задан", retryable=False)

    monkeypatch.setattr(main, "get_client", no_key)

    response = TestClient(main.app).post("/agent/ask", json={"question": "x"})

    assert response.status_code == 502
