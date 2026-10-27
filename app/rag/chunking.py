"""Этап 1: нарезка документов на куски с перекрытием."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    document: str
    index: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def split_text(document: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Нарезать `text` на куски примерно по `size` символов с перекрытием `overlap`.

    TODO(этап 1): начать с нарезки фиксированного размера, затем попробовать резать
    сначала по заголовкам и абзацам (рекурсивная нарезка) и сравнить качество
    поиска на этапе 7.
    """
    raise NotImplementedError
