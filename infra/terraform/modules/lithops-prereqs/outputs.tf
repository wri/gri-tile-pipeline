output "bucket_name" {
  description = "Lithops state bucket name for this region."
  value       = aws_s3_bucket.state.id
}

output "bucket_arn" {
  value = aws_s3_bucket.state.arn
}

output "region" {
  value = var.region
}
