import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings


def connect() -> psycopg.Connection:
    """Новое подключение к Postgres. Транзакции открываем явно через `conn.transaction()`."""
    conn = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    register_vector(conn)  # list[float] <-> vector без ручной сериализации
    return conn


def pgvector_version() -> str | None:
    """Версия установленного pgvector или None, если расширения нет."""
    with connect() as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return row[0] if row else None
