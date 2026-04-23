variable "bucket_name" {
  description = "Name of the bucket in the *other* account that this module will attach a policy to (e.g. tof-output in the wri account)."
  type        = string
}

variable "grantee_role_arns" {
  description = "IAM role ARNs that need read/write on the bucket. Typically the Lithops Lambda execution role in the compute account."
  type        = list(string)
}

variable "key_prefixes" {
  description = "Optional object-key prefixes to scope the grant to. Empty list means the whole bucket."
  type        = list(string)
  default     = []
}
