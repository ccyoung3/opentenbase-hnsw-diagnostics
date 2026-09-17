\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS hnsw_smoke;
CREATE TABLE hnsw_smoke (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    embedding vector(3) NOT NULL
);

INSERT INTO hnsw_smoke (embedding)
VALUES
    ('[1,0,0]'),
    ('[0,1,0]'),
    ('[0,0,1]'),
    ('[1,1,0]'),
    ('[1,1,1]');

CREATE INDEX hnsw_smoke_embedding_idx
ON hnsw_smoke USING hnsw (embedding vector_l2_ops);

SET enable_seqscan = off;

EXPLAIN (COSTS OFF)
SELECT id
FROM hnsw_smoke
ORDER BY embedding <-> '[1,0,0]'
LIMIT 3;

SELECT id, embedding <-> '[1,0,0]' AS distance
FROM hnsw_smoke
ORDER BY embedding <-> '[1,0,0]'
LIMIT 3;
