output "execution_role_arn" {
  description = "Lambda execution role ARN. Feed into Lithops aws_lambda.execution_role."
  value       = module.role.role_arn
}

output "lambda_role_arn" {
  description = "Alias for execution_role_arn. Used by docs/manual_wri_policy_update.md."
  value       = module.role.role_arn
}

output "state_bucket_usw2" {
  value = module.state_usw2.bucket_name
}

output "state_bucket_euc1" {
  value = module.state_euc1.bucket_name
}

output "state_bucket_use1" {
  value = module.state_use1.bucket_name
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "cross_account_policy_statements_json" {
  description = "Statement[] array (JSON-encoded) to append to tof-output's bucket policy. Consumed by scripts/merge_tof_output_policy.py."
  value       = module.cross_account_statements.statements_json
}

output "cross_account_policy_full_json" {
  description = "Reference-only: full IAM policy document for the cross-account grant. Not applied."
  value       = module.cross_account_statements.policy_json
}

data "aws_caller_identity" "current" {}
