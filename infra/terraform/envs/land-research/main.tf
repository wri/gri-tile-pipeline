module "state_usw2" {
  source = "../../modules/lithops-prereqs"
  region = "us-west-2"
  tags   = var.tags
}

module "state_euc1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.euc1
  }
  region = "eu-central-1"
  tags   = var.tags
}

module "state_use1" {
  source = "../../modules/lithops-prereqs"
  providers = {
    aws = aws.use1
  }
  region = "us-east-1"
  tags   = var.tags
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

# Cross-account grant is applied MANUALLY, This module only generates
# the statement JSON; it creates no AWS resources.
module "cross_account_statements" {
  source            = "../../modules/cross-account-s3-access"
  bucket_name       = var.output_bucket_name
  grantee_role_arns = [module.role.role_arn]
  key_prefixes      = var.output_bucket_prefixes
}
