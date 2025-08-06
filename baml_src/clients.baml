client<llm> Azure_openai  {
  provider azure-openai
  retry_policy Exponential
  options {
    base_url env.PROMPT_ENDPOINT
    api_version env.AZURE_OPENAI_API_VERSION
    api_key env.AZURE_OPENAI_API_KEY
    temperature 0.8
  }
}

retry_policy Exponential {
  max_retries 3
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}