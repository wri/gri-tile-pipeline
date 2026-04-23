# Output keys match envs/land-research/ so infra/lithops/render.py can
# substitute values into the same templates via `make -C infra render
# ENV=datalab-test`.

output "execution_role_arn" {
  description = "Lambda execution role ARN."
  value       = module.role.role_arn
}

output "lambda_role_arn" {
  description = "Alias for execution_role_arn (matches land-research output shape)."
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
  description = "Statement[] JSON. Not applied to tof-output in this env — for render exercise only."
  value       = module.cross_account_statements.statements_json
}

output "cross_account_policy_full_json" {
  value = module.cross_account_statements.policy_json
}

data "aws_caller_identity" "current" {}
