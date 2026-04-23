# Infrastructure

This directory holds all AWS infrastructure code for the GRI tile pipeline.

## Ownership model

Terraform owns the minimum. Lithops keeps ownership of its ECR repos, container
images, and Lambda functions — we deploy exclusively via `lithops runtime build`
and `lithops runtime deploy`.

| Resource                                    | Owner                   |
| ------------------------------------------- | ----------------------- |
| Lambda execution role (IAM)                 | Terraform               |
| Lithops state S3 buckets (one per region)   | Terraform               |
| Cross-account S3 access on `tof-output`     | Terraform (wri account) |
| Terraform state backend                     | Terraform bootstrap     |
| ECR repositories                            | Lithops                 |
| Container images                            | Lithops (`runtime build`) |
| Lambda functions                            | Lithops (`runtime deploy` or lazy) |
| Lithops configs (`.lithops/<env>/*.yaml`)   | `make render`           |

## Layout

```
infra/
├── terraform/
│   ├── bootstrap/                     # One-shot: TF state bucket + DynamoDB lock
│   ├── modules/
│   │   ├── lithops-iam-role/          # Lambda execution role
│   │   ├── lithops-prereqs/           # Per-region state bucket
│   │   └── cross-account-s3-access/   # tof-output policy in wri account
│   └── envs/
│       └── land-research/             # Wires the modules together
├── lithops/
│   ├── config.*.yaml.tmpl             # Templates rendered from Terraform outputs
│   └── render.py
└── Makefile                           # render + lithops build + lithops deploy
```

## Prerequisites

- Terraform >= 1.5
- AWS CLI v2 configured with credentials for:
  - The **land-research** account (admin for initial apply)
  - The **wri** account (admin on the `tof-output` bucket policy)
- Lithops installed in the local venv (`uv sync --extra loaders --extra predict`)
- Docker running locally (Lithops uses it to build container images)

## One-time bootstrap

Run once per new AWS account. Creates the Terraform state bucket and DynamoDB
lock table so all subsequent applies use remote state.

```bash
cd infra/terraform/bootstrap
terraform init
terraform apply
```

Capture the outputs; they feed the backend config for `envs/land-research`.

## Deploy the env

```bash
cd infra/terraform/envs/land-research
terraform init
terraform apply
```

This creates the IAM execution role, per-region Lithops state S3 buckets, and
the cross-account bucket policy on `tof-output` (needs wri credentials via
provider alias).

## Build and deploy Lithops runtimes

From the repo root:

```bash
# Render Lithops configs from Terraform outputs into .lithops/land-research/
make -C infra render ENV=land-research

# Build each runtime (Lithops creates ECR repo + pushes image)
make -C infra build-predict ENV=land-research
make -C infra build-loaders-usw2 ENV=land-research
make -C infra build-loaders-euc1 ENV=land-research
make -C infra build-s1 ENV=land-research

# Or all at once:
make -C infra build-all ENV=land-research
```

Switch the pipeline to the new account for a run:

```bash
LITHOPS_ENV=land-research gri-ttc predict tiles.csv --dest s3://tof-output
```

