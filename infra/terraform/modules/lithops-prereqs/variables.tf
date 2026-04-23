variable "region" {
  description = "AWS region this state bucket serves. Used to derive the default bucket name."
  type        = string
}

variable "bucket_name" {
  description = "Override the bucket name. Defaults to \"lithops-ttc-<region>\"."
  type        = string
  default     = null
}

variable "expiration_days" {
  description = "How long Lithops job state should live before expiring."
  type        = number
  default     = 7
}

variable "tags" {
  description = "Tags to apply to the bucket."
  type        = map(string)
  default     = {}
}
