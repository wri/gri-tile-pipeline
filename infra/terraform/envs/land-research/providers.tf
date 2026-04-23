terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state; backend values come from `terraform init -backend-config=...`
  # or the bootstrap stack's `backend_snippet` output.
  backend "s3" {
    key     = "envs/land-research/terraform.tfstate"
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

# Same account, us-east-1 (predict Lambda co-locates with tof-output).
provider "aws" {
  alias  = "use1"
  region = "us-east-1"
  default_tags {
    tags = local.default_aws_tags
  }
}
