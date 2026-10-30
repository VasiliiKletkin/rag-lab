-- Этап 2: векторы кусков.
-- NULL = ещё не посчитан. Модель хранится рядом с вектором: векторы разных моделей
-- несравнимы, даже если размерность совпадает.
ALTER TABLE chunks
    ADD COLUMN embedding vector(384),
    ADD COLUMN embedding_model text,
    ADD CONSTRAINT chunks_embedding_with_model
        CHECK ((embedding IS NULL) = (embedding_model IS NULL));

-- HNSW по косинусному расстоянию: оператор <=> в запросах. Строится на пустой таблице.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
