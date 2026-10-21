output "execution_role_arn" {
  description = "Lambda execution role ARN. Feed into Lithops aws_lambda.execution_role."
  value       = module.role.role_arn
}

output "lambda_role_arn" {
  description = "Alias for execution_role_arn."
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

output "ttc_bucket_name" {
  description = "Name of the dedicated TTC data bucket (ARD + predictions)."
  value       = aws_s3_bucket.ttc_data.id
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

data "aws_caller_identity" "current" {}
