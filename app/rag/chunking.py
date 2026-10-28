"""Этап 1: нарезка документов на куски с перекрытием."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    document: str
    index: int
    text: str
    start: int  # позиция начала куска в исходном тексте
    end: int  # позиция сразу после конца куска


def split_text(document: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Нарезать `text` на куски по `size` символов с перекрытием `overlap`.

    Скользящее окно: каждый следующий кусок начинается на `size - overlap` символов позже.
    Останавливаемся, как только окно дошло до конца текста, иначе в конце появится
    хвостик, целиком лежащий внутри предыдущего куска.
    """
    if size <= 0:
        raise ValueError("size должен быть больше 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap должен быть не меньше 0 и меньше size")

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(document, len(chunks), text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


# TODO(этап 1, часть 2): рекурсивная нарезка — сначала по заголовкам, потом по абзацам,
#   строкам и предложениям; не резать внутри блоков кода. Сравнить с фиксированной на этапе 7.
