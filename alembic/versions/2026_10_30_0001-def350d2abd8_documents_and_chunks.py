"""Документы, куски и векторы (этапы 1–2).

Revision ID: def350d2abd8
Revises:
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "def350d2abd8"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("chunker", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="documents_content_hash_sha256"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        # NULL = вектор ещё не посчитан; модель хранится рядом с вектором
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=True),
        sa.CheckConstraint("chunk_index >= 0", name="chunks_chunk_index_check"),
        sa.CheckConstraint("content <> ''", name="chunks_content_check"),
        sa.CheckConstraint("char_start >= 0", name="chunks_char_start_check"),
        sa.CheckConstraint("char_end > char_start", name="chunks_char_range_check"),
        sa.CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)", name="chunks_embedding_with_model"
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="chunks_document_id_chunk_index_key"
        ),
    )
    # HNSW по косинусному расстоянию (оператор <=>); строится и на пустой таблице
    op.create_index(
        "chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("chunks_embedding_hnsw", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    # расширение не удаляем: им могут пользоваться другие таблицы
