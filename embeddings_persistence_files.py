import os
import json
import logging

logger = logging.getLogger(__name__)


class EmbeddingsPersistence:
    def __init__(self, chunk_output_path: str):
        self.chunk_output_path = chunk_output_path

    def save_embeddings(self, embeddings: list):
        try:
            with open(self.chunk_output_path, "w") as f:
                json.dump(embeddings, f, indent=2)
        except Exception as e:
            logger.exception(
                f"Error saving embeddings to {self.chunk_output_path}: {e}"
            )
            return 0
        return len(embeddings)

    def load_embeddings(self):
        with open(self.chunk_output_path, "r") as f:
            chunks = json.load(f)
        return chunks
