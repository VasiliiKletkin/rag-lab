"""Этап 9: те же инструменты, что у агента, но по протоколу MCP.

Агент из этапа 8 знает инструменты, потому что мы передали их в своём коде. MCP переворачивает
это: сервер сам объявляет инструменты и ресурсы, а любой клиент (Claude Code, Claude Desktop,
Inspector) подключается и получает их список — без нашего кода на стороне клиента.

Запуск:  uv run python -m app.mcp_server            # stdio — клиент запускает нас как процесс
         uv run python -m app.mcp_server --http     # Streamable HTTP на 127.0.0.1:8001/mcp
"""

import argparse
import json
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from pydantic import Field, ValidationError

from app.db import connect
from app.rag import tools
from app.rag.embeddings import get_embedder

server = MCPServer(
    "rag-lab",
    instructions=(
        "Поиск по корпусу технической документации (pgvector, PostgreSQL, Python). "
        "Корпус на английском: формулируй запросы к search_docs по-английски."
    ),
)


# Схему инструмента SDK строит из сигнатуры: Annotated/Field попадают в JSON Schema,
# а ToolError SDK отдаёт клиенту текстом с is_error — как наш агент на этапе 8.
# Любое другое исключение он прячет за «Error executing tool», и модель не узнает причину.


@server.tool()
def search_docs(
    query: Annotated[str, Field(min_length=1, description="Запрос на английском")],
    k: Annotated[int, Field(ge=1, le=10, description="Сколько кусков вернуть")] = 5,
    mode: Literal["vector", "hybrid", "rerank"] = "rerank",
) -> list[dict]:
    """Найти куски текста по смыслу запроса. Возвращает id куска, документ, оценку и текст."""
    try:
        args = tools.SearchDocsArgs(query=query, k=k, mode=mode)
    except ValidationError as exc:
        raise ToolError(f"неверные аргументы: {exc}") from exc
    with connect() as conn:
        return tools.search_docs(conn, get_embedder(), args)


@server.tool()
def get_chunk(id: Annotated[int, Field(description="id куска из search_docs")]) -> list[dict]:
    """Кусок по id вместе с соседями из того же документа — когда найденный кусок обрывается."""
    with connect() as conn:
        try:
            return tools.get_chunk(conn, get_embedder(), tools.GetChunkArgs(id=id))
        except tools.ToolError as exc:
            raise ToolError(str(exc)) from exc


@server.tool()
def list_documents() -> list[dict]:
    """Документы корпуса: путь, заголовок, число кусков."""
    with connect() as conn:
        return tools.list_documents(conn, get_embedder(), tools.NoArgs())


# Ресурс — это данные, которые клиент читает сам (без решения модели), в отличие от инструмента.
@server.resource("rag://documents", mime_type="application/json")
def documents_resource() -> str:
    return json.dumps(list_documents(), ensure_ascii=False)


DOCUMENT_SQL = """
SELECT d.path, string_agg(c.content, E'\\n' ORDER BY c.chunk_index)
FROM documents d JOIN chunks c ON c.document_id = d.id
WHERE d.id = %(id)s GROUP BY d.path
"""


@server.resource("rag://documents/{document_id}", mime_type="text/plain")
def document_resource(document_id: int) -> str:
    """Весь документ, склеенный из кусков (с повторами на перекрытиях — это честно про хранение)."""
    with connect() as conn:
        row = conn.execute(DOCUMENT_SQL, {"id": document_id}).fetchone()
    if row is None:
        raise ResourceNotFoundError(f"документа с id={document_id} нет")
    return f"# {row[0]}\n\n{row[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-сервер rag-lab")
    parser.add_argument("--http", action="store_true", help="Streamable HTTP вместо stdio")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.http:
        server.run("streamable-http", port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
