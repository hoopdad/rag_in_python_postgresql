class WorkflowAnalysisDetails {
  logs string[] @description(#"
  Chunks of logs deemed by a vector proximity to be relevant to the question
  "#)
  question string @description(#"
  The key analysis question to consider when reviewing these log chunks
  "#)
}

class WorkflowCompletionStatus {
  result string  @description(#"
  Whether the final outcome of the log was successful. 
  Can be 'success', 'abort' or 'fail'
  "#)
  reason string @description(#"
  The reason that this was deemed an abort or a fail
  "#)
}

function DetermineWorkflowCompletionStatus(input: WorkflowAnalysisDetails) -> WorkflowCompletionStatus {
  client "Azure_openai"
  prompt #"
        You are a helpful DevOps engineer with expertise in Terraform, AWS and GitHub Actions. 
        Use the following log context to answer the question.
        Context:

        {% for logChunk in input.logs %}
        ----
        LOG CHUNK
        {{ logChunk }}
        ----
        {% endfor %}
    
          Question:
        {{input.question}}
        Give a one word error or warning reason that best summarizes. Respond only in JSON format

    {{ ctx.output_format }}
  "#
}

