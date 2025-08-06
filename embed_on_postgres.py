import dotenv
import os
import logging
import numpy as np
from openai import AzureOpenAI
from baml_client.sync_client import b as workflow_completion_client
from baml_client.types import (
    WorkflowCompletionStatus as workflow_completion_status,
    WorkflowAnalysisDetails,
)
from embeddings_persistence_postgres import EmbeddingsPersistencePostgres

logger = logging.getLogger(__name__)

# CONFIG for Embeddings
EMBEDDING_API_VERSION = "2024-12-01-preview"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
TOP_K = 5
CHUNK_OUTPUT_FILE = "log_chunks2.json"

# Config for Prompts
PROMPT_MODEL = "gpt-4o"


class AzureOpenAIEmbedding:
    def __init__(
        self,
        azure_endpoint: str,
        azure_model_name: str = EMBED_MODEL,
        azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY"),
        api_version: str = EMBEDDING_API_VERSION,
        chunk_output_path: str = CHUNK_OUTPUT_FILE,
        chunk_size: int = CHUNK_SIZE,
        chunk_top_k: int = TOP_K,
    ):
        self.azure_endpoint = azure_endpoint
        self.model_name = azure_model_name
        self.api_key = azure_openai_api_key
        self.api_version = api_version
        self.chunk_size = chunk_size
        self.chunk_output_path = chunk_output_path
        self.top_k = chunk_top_k
        self.chunk_saver = EmbeddingsPersistencePostgres(
            dsn=os.getenv(
                "POSTGRES_DSN", "postgresql://postgres:postgres@localhost/vector_db"
            )
        )
        self.embedding_client = self._get_embeddings_client()

    def get_top_chunks_for_question(self, log_file_name: str, question: str) -> list:
        try:
            q_embedding = self._get_embedding(question)
            q_embedding = q_embedding.tolist()  # Convert to list for JSON compatibility
            top_chunks = self.chunk_saver.cosine_similarity(
                file_name=log_file_name, question_vector=q_embedding
            )
        except Exception as e:
            logger.exception(f"Error getting context for chunks: {e}")
            raise e
        return top_chunks

    def embeddings_exist(self, file_name: str) -> bool:
        try:
            exists = self.chunk_saver.embeddings_exist(file_name=file_name)
        except Exception as e:
            logger.exception(f"Error checking if embeddings exist: {e}")
            raise e
        return exists

    def chunk_embed_and_save(self, log_path: str):
        logger.debug(f"Embedding and saving chunks from {log_path}")
        try:
            with open(log_path, "r") as f:
                text = f.read()
            chunks = self._chunk_text(text)
            results = []
            for chunk in chunks:
                emb = self._get_embedding(chunk)
                results.append({"chunk": chunk, "embedding": emb.tolist()})
            self.chunk_saver.save_embeddings(embeddings=results, file_name=log_path)
        except Exception as e:
            logger.exception(f"Error embedding chunks: {e}")
            return 0
        logger.debug(f"Saved {len(results)} chunks to {self.chunk_output_path}")

    def _get_embeddings_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

    def _get_embedding(self, text: str) -> np.array:
        try:
            if not isinstance(text, str):
                text = str(text)
            logger.debug("Getting embedding for text: %s", text[:50])
            response = self.embedding_client.embeddings.create(
                input=text, model=self.model_name
            )
            logger.debug(
                "Embeddings OpenAI API metadata:",
                {
                    "model": response.model,
                    "usage": response.usage,
                },
            )
            embedding_list: list[float] = response.data[0].embedding
            return np.array(embedding_list, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Error getting embedding: {e}")
            raise e

    def _chunk_text(self, text: str) -> list[str]:
        lines = text.splitlines()
        chunks = []

        max_chars = self.chunk_size  # e.g., 1500
        min_lines = 5  # Avoid chunks with too little context
        overlap = 3  # Number of lines to overlap

        i = 0
        while i < len(lines):
            current_chunk = []
            total_len = 0
            j = i

            while j < len(lines) and total_len + len(lines[j]) + 1 <= max_chars:
                current_chunk.append(lines[j])
                total_len += len(lines[j]) + 1
                j += 1

            if len(current_chunk) >= min_lines:
                chunks.append("\n".join(current_chunk).strip())

            # Slide the window forward with overlap
            i = j - overlap if j - overlap > i else j

        return chunks


def query_logs(question: str, log_file: str) -> str:

    embedding_client = AzureOpenAIEmbedding(
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        azure_model_name=os.getenv("AZURE_EMBEDDING_MODEL"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=EMBEDDING_API_VERSION,
        chunk_output_path=CHUNK_OUTPUT_FILE,
        chunk_size=CHUNK_SIZE,
        chunk_top_k=TOP_K,
    )

    if not embedding_client.embeddings_exist(file_name=log_file):
        logger.debug(f"Embeddings do not exist for {log_file}. Creating embeddings.")
        embedding_client.chunk_embed_and_save(log_file)  # Run once to create embeddings
    else:
        logger.debug(
            f"Embeddings already exist for {log_file}. Skipping embedding creation."
        )

    context = embedding_client.get_top_chunks_for_question(
        question=question, log_file_name=log_file
    )
    chunks = []
    for chnk in context:
        logger.debug(f"Chunk: {chnk['chunk'][:50]}... Score: {chnk['score']:.4f}")
        chunks.append(str(chnk["chunk"]))

    prompt_input: WorkflowAnalysisDetails = WorkflowAnalysisDetails(
        logs=chunks, question=question
    )
    response: workflow_completion_status = (
        workflow_completion_client.DetermineWorkflowCompletionStatus(input=prompt_input)
    )
    logger.debug(f"Response from workflow completion client: {response}")

    return response


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.basicConfig(
        level=logging.DEBUG,  # or INFO if you want less verbosity
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    logging.getLogger("embeddings_persistence_postgres").setLevel(logging.DEBUG)

    dotenv.load_dotenv()
    logger.debug(f"AZURE_EMBEDDING_ENDPOINT: {os.getenv('AZURE_EMBEDDING_ENDPOINT')}")
    logger.debug(f"PROMPT_ENDPOINT: {os.getenv('PROMPT_ENDPOINT')}")

    question = """
        Given all the warning and error conditions reported in this log, did this log succeed or fail? 
    """
    log_file = os.getenv("LOG_FILE")

    print(f"\nQuestion: ${question}")
    print("\nAnswer:")
    print(query_logs(question=question, log_file=log_file))
