# Working with Embeddings for RAG

This repo has two models: local files and processing, versus PostgreSQL.

- embed.py stores chunks and embeddings locally and implements a cosine similiarity function in python.
- embed_on_postgres.py stores chunks and embeddings in the database, and implements a cosine similiarity function in a postgresql/pgvector SELECT query.

This is configured to use an instance of OpenAI on Azure, but can be configured using BAML configurations and you might need alternate python libraries for embeddings.

## other files

- embddings_persistence_files.py directly does the write/read operations for vectors to files
- embeddings_persistence_postegres.py is the data layer, doing SELECT and INSERT operations as well as creating the table
- docker-compose.yml defines the configuration for a docker image managed by compose, to run a local database with persistent storage.
- requirements.txt - the python dependencies used in this project. install with `pip install -r requirements.txt`
- run-db.sh - a bash script to startup the database in docker compose

- baml_src/clients.baml configures the connection to our AI service
- baml_src/generators.baml defines information needed for the baml cli (not usually editing this one unless I change BAML version)
- baml_src/workflow_completion.baml defines the prompts and related data structures. 
- baml_client - these are generated files. Do not edit directly; run `baml-cli generate`

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
