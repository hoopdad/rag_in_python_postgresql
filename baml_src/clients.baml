client<llm> Azure_openai  {
  provider azure-openai
  options {
    base_url env.PROMPT_ENDPOINT
    api_version env.AZURE_OPENAI_API_VERSION
    api_key env.AZURE_OPENAI_API_KEY
    temperature 0.8
  }
}
