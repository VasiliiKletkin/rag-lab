-- Этап 1: документы и их куски.

CREATE TABLE documents (
    id            bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path          text        NOT NULL UNIQUE,
    title         text        NOT NULL,
    url           text,
    content_hash  text        NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    chunker       text        NOT NULL,  -- настройки нарезки, например 'fixed-800-120'
    ingested_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id           bigint  GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  bigint  NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  integer NOT NULL CHECK (chunk_index >= 0),
    content      text    NOT NULL CHECK (content <> ''),
    char_start   integer NOT NULL CHECK (char_start >= 0),
    char_end     integer NOT NULL CHECK (char_end > char_start),
    -- индекс начинается с document_id, поэтому покрывает и внешний ключ
    UNIQUE (document_id, chunk_index)
);
