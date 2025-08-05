class WorkflowCompletionStatus {
  result string
  reason string
}

function DetermineWorkflowCompletionStatus(context: string, question: string) -> WorkflowCompletionStatus {
  client "Azure_openai"
  prompt #"
        You are a helpful DevOps engineer with expertise in Terraform, AWS and GitHub Actions. 
        Use the following log context to answer the question.
        Context:
        {{context}}
        Question:
        {{question}}
        Give a one word error or warning reason that best summarizes. Respond only in JSON format

    {{ ctx.output_format }}
  "#
}

