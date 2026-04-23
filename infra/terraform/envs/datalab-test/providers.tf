terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # No backend block — local state. This env is throwaway (dl-user account
  # exercise while land-research perms are pending). Teardown is
  # `terraform destroy` + removing this directory.
}

locals {
  default_aws_tags = {
    "wri:project" = "gri-ttc-tile-generator-dltest"
    "ManagedBy"   = "terraform"
    "Ephemeral"   = "true"
  }
}

provider "aws" {
  region = "us-west-2"
  default_tags {
    tags = local.default_aws_tags
  }
}

provider "aws" {
  alias  = "euc1"
  region = "eu-central-1"
  default_tags {
    tags = local.default_aws_tags
  }
}

provider "aws" {
  alias  = "use1"
  region = "us-east-1"
  default_tags {
    tags = local.default_aws_tags
  }
}
