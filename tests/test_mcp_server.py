import json

import psycopg
import pytest
from mcp import Client

import app.mcp_server as mcp_server
from tests.test_retrieval import LengthEmbedder, corpus  # noqa: F401


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def wired(corpus: psycopg.Connection, monkeypatch):  # noqa: F811
    """Сервер смотрит в тестовую базу и использует быстрый эмбеддер вместо модели."""

    class Conn:
        def __enter__(self):
            return corpus

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(mcp_server, "connect", lambda: Conn())
    monkeypatch.setattr(mcp_server, "get_embedder", lambda: LengthEmbedder())
    return corpus


@pytest.mark.anyio
async def test_server_declares_tools_and_resources(wired) -> None:
    async with Client(mcp_server.server) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()

    assert {t.name for t in tools.tools} == {"search_docs", "get_chunk", "list_documents"}
    search = next(t for t in tools.tools if t.name == "search_docs")
    assert search.input_schema["required"] == ["query"]  # k и mode со значениями по умолчанию
    assert [str(r.uri) for r in resources.resources] == ["rag://documents"]
    assert [t.uri_template for t in templates.resource_templates] == [
        "rag://documents/{document_id}"
    ]


@pytest.mark.anyio
async def test_search_docs_over_mcp(wired) -> None:
    async with Client(mcp_server.server) as client:
        result = await client.call_tool("search_docs", {"query": "HNSW index", "mode": "hybrid"})

    assert result.is_error is False
    hits = json.loads(result.content[0].text)
    assert hits[0]["document"] == "pg.md"


@pytest.mark.anyio
async def test_tool_errors_are_results_not_crashes(wired) -> None:
    async with Client(mcp_server.server) as client:
        missing = await client.call_tool("get_chunk", {"id": 999999})
        bad_k = await client.call_tool("search_docs", {"query": "x", "k": 500})

    assert missing.is_error is True and "999999" in missing.content[0].text
    assert bad_k.is_error is True


@pytest.mark.anyio
async def test_document_resource_returns_whole_text(wired) -> None:
    doc_id = wired.execute("SELECT id FROM documents WHERE path = 'py.md'").fetchone()[0]
    async with Client(mcp_server.server) as client:
        listing = await client.read_resource("rag://documents")
        document = await client.read_resource(f"rag://documents/{doc_id}")

    assert {d["path"] for d in json.loads(listing.contents[0].text)} == {"pg.md", "py.md"}
    assert document.contents[0].text.startswith("# py.md\n\nUse append")
