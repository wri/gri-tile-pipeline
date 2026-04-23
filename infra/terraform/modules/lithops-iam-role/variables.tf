variable "name" {
  description = "Name for the Lambda execution role."
  type        = string
  default     = "lithops-execution-role"
}

variable "lithops_state_bucket_arns" {
  description = "ARNs of the Lithops state S3 buckets this role can read/write."
  type        = list(string)
}

variable "output_bucket_arn" {
  description = "ARN of the pipeline output bucket (e.g. arn:aws:s3:::tof-output in the wri account)."
  type        = string
}

variable "tags" {
  description = "Tags to apply to the role."
  type        = map(string)
  default     = {}
}
