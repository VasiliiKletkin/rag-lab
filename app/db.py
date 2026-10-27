import psycopg

from app.config import get_settings


def pgvector_version() -> str | None:
    """Версия установленного pgvector или None, если расширения нет."""
    with psycopg.connect(get_settings().database_url, connect_timeout=3) as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return row[0] if row else None
