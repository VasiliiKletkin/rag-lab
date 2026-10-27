"""Stage 2: turn text into vectors."""

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# TODO(stage 2): LocalEmbedder backed by sentence-transformers (settings.embedding_model),
#   optionally an API-backed embedder (e.g. Voyage AI) behind the same protocol.
