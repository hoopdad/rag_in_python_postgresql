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
        Question:
        {{input.question}}

        Context:

        {% for logChunk in input.logs %}
        ----
        LOG CHUNK
        {{ logChunk }}
        ----
        {% endfor %}
    
    {{ ctx.output_format }}
  "#
}

