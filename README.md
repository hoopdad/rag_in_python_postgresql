# Working with Embeddings for RAG

This repo has two models: local files and processing, versus PostgreSQL.

- embed.py stores chunks and embeddings locally and implements a cosine similiarity function in python.
- embed_on_postgres.py stores chunks and embeddings in the database, and implements a cosine similiarity function in a postgresql/pgvector SELECT query.

## for the database version

### Setup database

```bash
docker compose up -d
```

### Setup Environment

Set values for all these in a file called `.env`

```bash
AZURE_OPENAI_API_KEY="<your API key>"
LOG_FILE="<log file to analyze>"
AZURE_EMBEDDING_ENDPOINT="<your azure endpoint for embeddings>"
AZURE_EMBEDDING_MODEL="<your embedding model from azure>"
PROMPT_ENDPOINT="<your azure endpoint for prompts>"
AZURE_OPENAI_API_VERSION="<Azure Open AI API Version>"
```

### Run the program

```bash
python3 ./embed_on_postgres.py
```
## for the local files version

### Setup Environment

Set values for all these in a file called `.env`

```bash
AZURE_OPENAI_API_KEY="<your API key>"
LOG_FILE="<log file to analyze>"
AZURE_EMBEDDING_ENDPOINT="<your azure endpoint for embeddings>"
AZURE_EMBEDDING_MODEL="<your embedding model from azure>"
PROMPT_ENDPOINT="<your azure endpoint for prompts>"
AZURE_OPENAI_API_VERSION="<Azure Open AI API Version>"
```

### Run the program

```bash
python3 ./embed.py
```
