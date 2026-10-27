"""Stage 4: answer a question from retrieved chunks with Claude."""

from app.schemas import AskResponse, SearchHit


def answer(question: str, hits: list[SearchHit]) -> AskResponse:
    """TODO(stage 4): send each hit as a `document` content block with citations enabled,
    model = settings.llm_model; map the returned citations back to chunk ids.
    Answer "I don't know" when the context does not contain the answer."""
    raise NotImplementedError
