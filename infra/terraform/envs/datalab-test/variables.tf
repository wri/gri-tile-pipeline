variable "role_name" {
  description = "Name for the Lambda execution role. Suffixed to avoid colliding with any pre-existing Lithops role in datalab."
  type        = string
  default     = "lithops-execution-role-dltest"
}

variable "bucket_prefix" {
  description = "Prefix for the three per-region Lithops state buckets. Combined with the region to produce globally-unique names."
  type        = string
  default     = "gri-ttc-dltest-state"
}

variable "output_bucket_name" {
  description = "Pipeline output bucket. Datalab-test does not actually grant to it; this is exercised for output-generation only."
  type        = string
  default     = "tof-output"
}

variable "output_bucket_prefixes" {
  description = "Restrict the cross-account grant to these prefixes. Empty list = whole bucket."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to resources that thread var.tags through (state buckets, role). Merged on top of provider default_tags."
  type        = map(string)
  default = {
    Project   = "gri-tile-pipeline-dltest"
    ManagedBy = "terraform"
    Ephemeral = "true"
  }
}
