module "state_usw2" {
  source      = "../../modules/lithops-prereqs"
  region      = "us-west-2"
  bucket_name = "${var.bucket_prefix}-us-west-2"
  tags        = var.tags
}

module "state_euc1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.euc1
  }
  region      = "eu-central-1"
  bucket_name = "${var.bucket_prefix}-eu-central-1"
  tags        = var.tags
}

module "state_use1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.use1
  }
  region      = "us-east-1"
  bucket_name = "${var.bucket_prefix}-us-east-1"
  tags        = var.tags
}

module "role" {
  source = "../../modules/lithops-iam-role"
  name   = var.role_name
  lithops_state_bucket_arns = [
    module.state_usw2.bucket_arn,
    module.state_euc1.bucket_arn,
    module.state_use1.bucket_arn,
  ]
  output_bucket_arn = "arn:aws:s3:::${var.output_bucket_name}"
  tags              = var.tags
}

# Output-only; no AWS resource created. Exercised for parity with the
# real env — confirms the module compiles + emits statement JSON.
module "cross_account_statements" {
  source            = "../../modules/cross-account-s3-access"
  bucket_name       = var.output_bucket_name
  grantee_role_arns = [module.role.role_arn]
  key_prefixes      = var.output_bucket_prefixes
}
