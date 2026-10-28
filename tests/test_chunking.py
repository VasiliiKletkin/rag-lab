import pytest

from app.rag.chunking import split_text

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def test_overlap_between_neighbours() -> None:
    chunks = split_text("doc.md", ALPHABET, size=10, overlap=3)

    assert [c.text for c in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_empty_text_gives_no_chunks() -> None:
    assert split_text("doc.md", "", size=10, overlap=3) == []


def test_short_text_is_one_chunk() -> None:
    chunks = split_text("doc.md", "abc", size=10, overlap=3)

    assert [c.text for c in chunks] == ["abc"]


def test_no_duplicate_tail_when_window_hits_the_end() -> None:
    chunks = split_text("doc.md", ALPHABET[:24], size=10, overlap=3)

    assert [c.text for c in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx"]


@pytest.mark.parametrize(("size", "overlap"), [(10, 0), (10, 3), (7, 6), (100, 20)])
def test_chunks_glue_back_into_original(size: int, overlap: int) -> None:
    chunks = split_text("doc.md", ALPHABET * 5, size=size, overlap=overlap)

    glued = chunks[0].text + "".join(c.text[overlap:] for c in chunks[1:])
    assert glued == ALPHABET * 5


def test_positions_and_indexes() -> None:
    text = ALPHABET * 3
    chunks = split_text("doc.md", text, size=10, overlap=3)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.document == "doc.md" for c in chunks)
    assert all(text[c.start : c.end] == c.text for c in chunks)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_settings_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("doc.md", ALPHABET, size=size, overlap=overlap)
