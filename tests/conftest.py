import os
from collections.abc import Iterator

import psycopg
import pytest
from alembic import command
from alembic.config import Config

ADMIN_URL = os.environ.get("TEST_DATABASE_ADMIN_URL", "postgresql://rag:rag@localhost:5432/rag")
TEST_DB = "rag_test"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Отдельная база rag_test, пересоздаётся на каждый прогон тестов.

    Рабочую базу тесты не трогают: загрузка удаляет документы, которых нет в списке,
    и на рабочей базе стёрла бы корпус. Без Postgres тесты пропускаются, а в CI
    (REQUIRE_DB=1) падают, чтобы не пройти незаметно.
    """
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        if os.environ.get("REQUIRE_DB"):
            raise
        pytest.skip("Postgres недоступен")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB
    # схема тестовой базы — те же миграции Alembic, что и у рабочей
    os.environ["DATABASE_URL"] = url
    from app.config import get_settings

    get_settings.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")
    return url


@pytest.fixture
def db(test_db_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        yield conn
