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
  description = "ARN of the pipeline output bucket (same account; Lambda reads/writes ARD and predictions here)."
  type        = string
}

variable "additional_output_bucket_arns" {
  description = "Extra output bucket ARNs the Lambda role can read/write (e.g. the orchestration repo's wri-restoration-geodata bucket where sentinel-flow stages tiles)."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to the role."
  type        = map(string)
  default     = {}
}
