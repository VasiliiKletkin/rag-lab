from alembic import command
from alembic.config import Config


def test_models_match_migrations(test_db_url: str) -> None:
    """После upgrade head модели в app/models.py не должны расходиться с базой.

    Иначе кто-то поменял модель и забыл `alembic revision --autogenerate`.
    """
    command.check(Config("alembic.ini"))  # бросает исключение, если есть расхождения
