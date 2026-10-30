"""Схема базы в стиле SQLAlchemy 2.0. Источник правды для миграций Alembic."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    chunker: Mapped[str] = mapped_column(Text)  # настройки нарезки, например 'fixed-800-120'
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="documents_content_hash_sha256"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # NULL = вектор ещё не посчитан. Модель хранится рядом: векторы разных моделей
    # несравнимы, даже если размерность совпадает.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        # индекс начинается с document_id, поэтому покрывает и внешний ключ
        UniqueConstraint("document_id", "chunk_index", name="chunks_document_id_chunk_index_key"),
        CheckConstraint("chunk_index >= 0", name="chunks_chunk_index_check"),
        CheckConstraint("content <> ''", name="chunks_content_check"),
        CheckConstraint("char_start >= 0", name="chunks_char_start_check"),
        CheckConstraint("char_end > char_start", name="chunks_char_range_check"),
        CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)", name="chunks_embedding_with_model"
        ),
        # HNSW по косинусному расстоянию: оператор <=> в запросах
        Index(
            "chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
