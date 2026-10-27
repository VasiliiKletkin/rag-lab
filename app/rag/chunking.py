"""Stage 1: split documents into overlapping chunks."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    document: str
    index: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def split_text(document: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split `text` into chunks of about `size` characters with `overlap` between neighbours.

    TODO(stage 1): start with a fixed-size splitter, then try splitting on headings and
    paragraphs first (recursive splitting) and compare retrieval quality in stage 7.
    """
    raise NotImplementedError
