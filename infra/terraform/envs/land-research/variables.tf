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

variable "tags" {
  description = "Tags applied to everything this stack creates."
  type        = map(string)
  default = {
    Project   = "gri-tile-pipeline"
    ManagedBy = "terraform"
  }
}
