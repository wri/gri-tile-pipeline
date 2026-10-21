# Infrastructure

This directory holds all AWS infrastructure code for the GRI tile pipeline.

## Ownership model

Single AWS account: **land-research** (`058755926933`). Terraform owns the
minimum — Lithops keeps ownership of its ECR repos, container images,
and Lambda functions, which we deploy via `lithops runtime build` and
`lithops runtime deploy`.

| Resource                                    | Owner                              |
| ------------------------------------------- | ---------------------------------- |
| Lambda execution role (IAM)                 | Terraform                          |
| Lithops state S3 buckets (one per region)   | Terraform                          |
| TTC data bucket (`wri-restoration-geodata-ttc`) | Terraform                      |
| Terraform state backend (shared)            | `gri-prefect-orchestration` workflow — this pipeline writes to key `gri-tile-pipeline/lr.tfstate` in `wri-restoration-terraform-state-lr` |
| ECR repositories                            | Lithops                            |
| Container images                            | Lithops (`runtime build`)          |
| Lambda functions                            | Lithops (`runtime deploy` or lazy) |
| Lithops configs (`.lithops/<env>/*.yaml`)   | `make render`                      |

## Layout

```
infra/
├── terraform/
│   ├── modules/
│   │   ├── lithops-iam-role/          # Lambda execution role
│   │   ├── lithops-prereqs/           # Per-region Lithops state bucket
│   │   └── cross-account-s3-access/   # Only used by legacy datalab-test env
│   └── envs/
│       ├── land-research/             # Primary: single-account, shared TF state
│       └── datalab-test/              # Legacy: two-account dev env, local state
├── lithops/
│   ├── config.*.yaml.tmpl             # Templates rendered from Terraform outputs
│   └── render.py
└── Makefile                           # render + lithops build + lithops deploy
```

## Prerequisites

- Terraform >= 1.5
- AWS CLI v2 with SSO profiles for land-research:
  - `AWSAdministratorAccess-058755926933` — for `terraform apply` and
    `lithops runtime build/deploy`
  - `LandResearchUser-058755926933` — day-to-day workflow runs (`gri-ttc`)
- Lithops installed in the local venv (`uv sync --extra loaders --extra predict`)
- Docker running locally, with the containerd image store disabled
  (Lambda only accepts Docker manifest v2)

## Deploy the env

```bash
cd infra/terraform/envs/land-research

AWS_PROFILE=AWSAdministratorAccess-058755926933 terraform init \
    -backend-config=bucket=wri-restoration-terraform-state-lr \
    -backend-config=region=us-east-1 \
    -backend-config=dynamodb_table=terraform-state-lock

AWS_PROFILE=AWSAdministratorAccess-058755926933 terraform apply
```

Creates the IAM execution role, the three per-region Lithops state
buckets, and the `wri-restoration-geodata-ttc` data bucket.

## Build and deploy Lithops runtimes

From the repo root:

```bash
# Render Lithops configs from Terraform outputs into .lithops/land-research/
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra render ENV=land-research

# Build each runtime (Lithops creates ECR repo + pushes image)
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-predict ENV=land-research
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-loaders-usw2 ENV=land-research
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-loaders-euc1 ENV=land-research
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-s1 ENV=land-research

# Or all at once:
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-all ENV=land-research
```

Drive the pipeline:

```bash
export AWS_PROFILE=AWSAdministratorAccess-058755926933   # or LandResearchUser-058755926933
export LITHOPS_ENV=land-research
gri-ttc run-project GHA_22_INEC --dest s3://wri-restoration-geodata-ttc --yes
```
