output "role_arn" {
  description = "ARN of the Lambda execution role. Feed this into Lithops aws_lambda.execution_role."
  value       = aws_iam_role.this.arn
}

output "role_name" {
  value = aws_iam_role.this.name
}
