variable "role_name" {
  description = "Name for the Lambda execution role created in land-research."
  type        = string
  default     = "lithops-execution-role"
}

variable "output_bucket_name" {
  description = "Pipeline output bucket in the wri account."
  type        = string
  default     = "tof-output"
}

variable "output_bucket_prefixes" {
  description = "Restrict the cross-account grant to these prefixes. Empty list = whole bucket."
  type        = list(string)
  # Current pipeline writes:
  #   {year}/raw/{X}/{Y}/raw/...   (ARD)
  #   {year}/tiles/{X}/{Y}/...     (predictions)
  default = []
}

variable "tags" {
  description = "Tags applied to everything this stack creates."
  type        = map(string)
  default = {
    Project   = "gri-tile-pipeline"
    ManagedBy = "terraform"
  }
}
