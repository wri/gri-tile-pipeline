output "policy_json" {
  description = "Full IAM policy document (Version + Statement[]). For reference/debugging; not applied."
  value       = data.aws_iam_policy_document.access.json
}

output "statements_json" {
  description = "Just the Statement[] array, JSON-encoded. Append to an existing bucket policy's Statement[] via scripts/merge_tof_output_policy.py."
  value       = jsonencode(jsondecode(data.aws_iam_policy_document.access.json).Statement)
}
