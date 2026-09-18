variable "role_name" {
  description = "Name for the Lambda execution role created in land-research."
  type        = string
  default     = "lithops-execution-role"
}

variable "output_bucket_name" {
  description = "Dedicated TTC pipeline output bucket (ARD + predictions), in this account, us-east-1."
  type        = string
  default     = "wri-restoration-geodata-ttc"
}

variable "additional_output_bucket_arns" {
  description = "Extra output bucket ARNs the Lambda role can write to. gri-prefect-orchestration's sentinel-batch-flow writes into wri-restoration-geodata; without that ARN here, Lambda PUTs return 403 and Lithops marks the jobs failed silently."
  type        = list(string)
  default = [
    "arn:aws:s3:::wri-restoration-geodata",
  ]
}

variable "tags" {
  description = "Tags applied to everything this stack creates."
  type        = map(string)
  default = {
    Project   = "gri-tile-pipeline"
    ManagedBy = "terraform"
  }
}
