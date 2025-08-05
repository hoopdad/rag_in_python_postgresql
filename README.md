#

## Setup database

```bash
docker compose up -d
```

## Setup Environment

Set values for all these in a file called `.env`

```bash
AZURE_OPENAI_API_KEY="<your API key>"
LOG_FILE="<log file to analyze>>"
AZURE_EMBEDDING_ENDPOINT="<your azure endpoint for embeddings>"
AZURE_EMBEDDING_MODEL="<your embedding model from azure>"
PROMPT_ENDPOINT="<your azure endpoint for prompts>"
AZURE_OPENAI_API_VERSION="<Azure Open AI API Version>"
```

## Run the program

```bash
python3 ./embed.py
```
