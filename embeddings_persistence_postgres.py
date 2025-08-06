import psycopg
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)


class EmbeddingsPersistencePostgres:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._ensure_table()

    def _ensure_table(self):
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    # Ensure pgvector extension is enabled
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # Ensure log_chunks table exists
                    cur.execute(
                        """
                        DROP TABLE IF EXISTS log_chunks;
                        CREATE TABLE IF NOT EXISTS log_chunks (
                            id SERIAL PRIMARY KEY,
                            file_name TEXT NOT NULL,
                            chunk TEXT NOT NULL,
                            embedding vector(1536)  -- Match embedding dimension
                        );
                    """
                    )
        except Exception as e:
            logger.exception("Failed to create table or extension.", e)
            raise

    def save_embeddings(self, file_name: str, embeddings: list[float]):
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    insert_query = "INSERT INTO log_chunks (file_name, chunk, embedding) VALUES (%s, %s, %s)"
                    values = []
                    for e in embeddings:
                        logger.error(f"Processing embedding: {str(json.dumps(e))}")
                        if (
                            not isinstance(e, dict)
                            or "embedding" not in e
                            or not isinstance(e["embedding"], list)
                            or len(e["embedding"]) != 1536
                        ):
                            logger.warning(f"Skipping invalid embedding")
                            continue
                        values.append((file_name, e["chunk"], e["embedding"]))
                    if values:
                        x = cur.executemany(insert_query, values)
                        logger.info(f"Saved {len(values)} embeddings. \n {x}")

        except Exception as e:
            logger.exception(f"Error saving embeddings to PostgreSQL: {e}", e)
            raise
        return len(embeddings)

    def load_embeddings(self, file_name: str) -> list:
        chunks = []
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    logger.debug(
                        f"Loading embeddings from PostgreSQL with file_name: {file_name}"
                    )
                    cur.execute(
                        "SELECT chunk, embedding FROM log_chunks WHERE file_name = %s",
                        (file_name,),
                    )
                    for row in cur.fetchall():
                        emb = row[1]
                        try:
                            emb = [float(x) for x in json.loads(emb)]
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Embedding contains non-numeric data: {e}")
                            continue
                        if len(emb) != 1536:
                            logger.warning(
                                f"Skipping embedding with incorrect shape. {len(emb)}"
                            )
                            continue
                        chunks.append(
                            {
                                "chunk": str(row[0]),
                                "embedding": list(
                                    emb
                                ),  # Convert to list for JSON compatibility
                            }
                        )
        except Exception as e:
            logger.exception(f"Error loading embeddings from PostgreSQL: {e}", e)
            raise
        return chunks
