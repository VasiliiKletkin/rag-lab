import anthropic
import httpx2 as httpx
import pytest
from anthropic.types import CitationCharLocation, Message, TextBlock, Usage

from app.rag.generation import NO_ANSWER, ClaudeGenerator, GenerationError
from app.schemas import SearchHit

HITS = [
    SearchHit(
        chunk_id="7", document="pg.md", text="Supported index types are HNSW and IVFFlat.", score=1
    ),
    SearchHit(chunk_id="9", document="py.md", text="Use append to push on the stack.", score=0.5),
]


def message(*blocks: TextBlock, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_1",
        model="claude-sonnet-5-5",
        role="assistant",
        type="message",
        content=list(blocks),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=20),
    )


def cited(text: str, document_index: int, quote: str) -> TextBlock:
    return TextBlock(
        type="text",
        text=text,
        citations=[
            CitationCharLocation(
                type="char_location",
                cited_text=quote,
                document_index=document_index,
                document_title=None,
                start_char_index=0,
                end_char_index=len(quote),
            )
        ],
    )


class FakeMessages:
    def __init__(self, result: Message | Exception) -> None:
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs) -> Message:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: Message | Exception) -> None:
        self.messages = FakeMessages(result)


def generator(result: Message | Exception) -> tuple[ClaudeGenerator, FakeClient]:
    client = FakeClient(result)
    return ClaudeGenerator(client, "claude-sonnet-5-5"), client  # type: ignore[arg-type]


def test_documents_and_question_are_sent_with_citations_enabled() -> None:
    gen, client = generator(message(TextBlock(type="text", text="HNSW и IVFFlat.", citations=None)))

    gen.answer("Какие индексы?", HITS)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[0]["source"]["data"] == HITS[0].text
    assert content[0]["citations"] == {"enabled": True}
    assert content[-1]["text"] == "Какие индексы?"


def test_citations_map_back_to_chunks() -> None:
    gen, _ = generator(
        message(
            TextBlock(type="text", text="pgvector поддерживает ", citations=None),
            cited("HNSW и IVFFlat.", document_index=0, quote="HNSW and IVFFlat"),
        )
    )

    result = gen.answer("Какие индексы?", HITS)

    assert result.answer == "pgvector поддерживает HNSW и IVFFlat."
    assert [(c.document, c.chunk_id, c.quote) for c in result.citations] == [
        ("pg.md", "7", "HNSW and IVFFlat")
    ]


def test_no_answer_has_no_citations() -> None:
    gen, _ = generator(message(cited(NO_ANSWER, document_index=1, quote="append")))

    result = gen.answer("Как приготовить борщ?", HITS)

    assert result.answer == NO_ANSWER
    assert result.citations == []


def test_no_hits_means_no_answer_without_api_call() -> None:
    gen, client = generator(message())

    assert gen.answer("x", []).answer == NO_ANSWER
    assert client.messages.calls == []


def test_refusal_is_not_retryable() -> None:
    gen, _ = generator(message(stop_reason="refusal"))

    with pytest.raises(GenerationError) as exc:
        gen.answer("x", HITS)
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (anthropic.RateLimitError, True),
        (anthropic.InternalServerError, True),
        (anthropic.BadRequestError, False),
    ],
)
def test_api_errors_become_generation_errors(error: type, retryable: bool) -> None:
    status = {anthropic.RateLimitError: 429, anthropic.InternalServerError: 500}.get(error, 400)
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com"))
    gen, _ = generator(error("boom", response=response, body=None))

    with pytest.raises(GenerationError) as exc:
        gen.answer("x", HITS)
    assert exc.value.retryable is retryable


@pytest.mark.parametrize("value", ["", None])
def test_missing_or_empty_key_is_reported(monkeypatch, value: str | None) -> None:
    from app.config import get_settings
    from app.rag.generation import get_client, get_generator

    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    monkeypatch.setattr("app.config.Settings.model_config", {"env_file": None, "extra": "ignore"})
    get_settings.cache_clear()
    get_generator.cache_clear()
    get_client.cache_clear()
    try:
        with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
            get_generator()
    finally:
        get_settings.cache_clear()
        get_generator.cache_clear()
        get_client.cache_clear()
