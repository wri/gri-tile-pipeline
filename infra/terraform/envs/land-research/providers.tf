terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state on the shared land-research state bucket.
  # Bucket / region / lock table come from `terraform init -backend-config=...`:
  #   bucket         = wri-restoration-terraform-state-lr
  #   region         = us-east-1
  #   dynamodb_table = terraform-state-lock
  backend "s3" {
    key     = "gri-tile-pipeline/lr.tfstate"
    encrypt = true
  }
}

locals {
  # Tags applied automatically to every resource any of these providers
  # creates. Resource-level `tags = var.tags` blocks still work and merge
  # with these; explicit tags win on key collision.
  default_aws_tags = {
    "wri:project" = "gri-ttc-tile-generator"
    "ManagedBy"   = "terraform"
  }
}

# Default provider: land-research, us-west-2 (S2 + S1 loaders live here).
# Caller supplies credentials via AWS_PROFILE or standard AWS env vars.
provider "aws" {
  region = "us-west-2"
  default_tags {
    tags = local.default_aws_tags
  }
}

# Same account, eu-central-1 (DEM loader co-locates with Copernicus).
provider "aws" {
  alias  = "euc1"
  region = "eu-central-1"
  default_tags {
    tags = local.default_aws_tags
  }
}

# Same account, us-east-1 (predict Lambda co-locates with the TTC data bucket).
provider "aws" {
  alias  = "use1"
  region = "us-east-1"
  default_tags {
    tags = local.default_aws_tags
  }
}
