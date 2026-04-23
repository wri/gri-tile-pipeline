# Setup guide for standing up the pipeline in the `land-research` AWS account

This is the linear runbook for a first-time setup while we straddle a couple cloud accounts. Follow the steps in order; each ends with a verification you should see pass before moving on.

**Two accounts, one direction:**

- **`land-research`** — compute (IAM role, Lithops state buckets, ECR,
  Lambda). Everything Terraform applies lands here. Credentials via the
  `<land_research_profile>` SSO profile.
- **`wri`** — the `tof-output` data bucket. Terraform never touches this
  account. The cross-account grant is applied manually in step 3, using
  the `<wri_acct_profile>` profile.

**Regions, by runtime:**

| Runtime | Region | Co-located with |
|---|---|---|
| S2 loader (`ttc-loaders-dev`) | us-west-2 | Sentinel-2 Registry of Open Data |
| DEM loader (`ttc-loaders-dev`) | eu-central-1 | Copernicus DEM |
| S1 RTC loader (`ttc-s1-dev`) | us-west-2 | (Azure, internet egress either way) |
| Predict (`ttc-predict-dev`) | us-east-1 | `tof-output` |

Already set up and just want to drive the CLI? Skip to
[`cli_workflows.md`](cli_workflows.md).

---

## Prerequisites

**Local tools**

- Terraform ≥ 1.5
- AWS CLI v2
- Docker (daemon running — `docker info` must succeed)
- `uv` for Python deps
- `jq` (used by the manual runbook in step 3)

**Python deps**

```bash
uv sync --extra loaders --extra predict --extra zonal --extra dev
# Or the bundle: uv sync --extra all
```

Verify:

```bash
uv run lithops --version         # should print 3.6.1
uv run gri-ttc --version         # should print the package version
```

**AWS profiles in `~/.aws/config`**

- `<land_research_profile>` — SSO profile for the land-research account. Used for all
  Terraform commands, Lithops builds, and every `gri-ttc` invocation.
- `<wri_acct_profile>` — profile for the wri account. Only used in step 3 (manual
  bucket-policy update). It can be any credential type the AWS SDK
  understands — SSO, IAM user keys, or assume-role — because it's invoked
  purely through the AWS CLI. The name is arbitrary; substitute yours.


---

## Step 0 — Verify credentials

```bash
# SSO login for <land_research_profile>:
aws sso login --profile <land_research_profile>

# If <wri_acct_profile> is also SSO (skip otherwise):
# aws sso login --profile <wri_acct_profile>

# Both must succeed:
aws sts get-caller-identity --profile <land_research_profile>   # should show land-research
aws sts get-caller-identity --profile <wri_acct_profile>   # should show wri
```

SSO tokens expire (~8h). If a later step fails with `ExpiredToken`, re-run
`aws sso login` for the affected profile.

---

## Step 1 — Bootstrap the Terraform state backend (one-shot)

Creates the S3 state bucket and DynamoDB lock table Terraform uses for
every subsequent apply.

```bash
AWS_PROFILE=<land_research_profile> terraform -chdir=infra/terraform/bootstrap init
AWS_PROFILE=<land_research_profile> terraform -chdir=infra/terraform/bootstrap apply
```

**Expected:** two resources (`gri-tile-pipeline-tfstate` S3 bucket and
`gri-tile-pipeline-tflock` DynamoDB table). Apply prints a `backend_snippet`
output — note it, you'll pass its values to `terraform init` in the next
step.

---

## Step 2 — Apply the land-research env

Creates the Lambda execution role and three Lithops state buckets (one per
region the pipeline uses) inside `land-research`. Emits the JSON
statements needed for the manual wri step as Terraform outputs. Does **not**
touch the wri account.

```bash
cd infra/terraform/envs/land-research
cp terraform.tfvars.example terraform.tfvars
# (optional) edit terraform.tfvars to scope output_bucket_prefixes

AWS_PROFILE=<land_research_profile> terraform init \
    -backend-config=bucket=gri-tile-pipeline-tfstate \
    -backend-config=region=us-east-1 \
    -backend-config=dynamodb_table=gri-tile-pipeline-tflock
AWS_PROFILE=<land_research_profile> terraform apply
```

**Expected:** apply succeeds without prompting for wri credentials.
Confirm the outputs are populated:

```bash
terraform -chdir=infra/terraform/envs/land-research output -raw lambda_role_arn
# arn:aws:iam::<account>:role/lithops-execution-role

terraform -chdir=infra/terraform/envs/land-research output -raw state_bucket_use1
# gri-tile-pipeline-lithops-state-us-east-1  (or similar)
```

---

## Step 3 — Apply the cross-account grant manually (wri account)

This is a separate, deliberate, per-human-hand step. Follow
[`manual_wri_policy_update.md`](manual_wri_policy_update.md) end to end.
Summary:

1. Snapshot the current `tof-output` policy to a local rollback file.
2. Export `cross_account_policy_statements_json` from step 2's outputs.
3. Run `scripts/merge_tof_output_policy.py` to merge without clobbering any
   pre-existing statements (the script dedupes by `Sid`, so re-running is
   safe).
4. `diff` and eyeball the merged policy — expect only additions.
5. `aws s3api put-bucket-policy` with `--profile <wri_acct_profile>`.
6. Keep the pre-apply snapshot file so you can roll back if needed.

**Expected:**

```bash
aws s3api get-bucket-policy --bucket tof-output --profile <wri_acct_profile> \
    | jq -r .Policy | jq '.Statement | length'
```

returns the pre-inventory count plus the two new cross-account statements
(`CrossAccountListBucket`, `CrossAccountReadWrite`). Re-run the runbook any
time `terraform apply` changes the role ARN or `output_bucket_prefixes`.

---

## Step 4 — Render Lithops configs

```bash
AWS_PROFILE=<land_research_profile> make -C infra render ENV=land-research
```

Substitutes Terraform outputs into `infra/lithops/*.yaml.tmpl`, writing:

- `.lithops/land-research/config.predict.yaml` (us-east-1)
- `.lithops/land-research/config.loaders-usw2.yaml` (us-west-2, S2)
- `.lithops/land-research/config.loaders-euc1.yaml` (eu-central-1, DEM)
- `.lithops/land-research/config.s1.yaml` (us-west-2, S1)

**Expected:** every `${VAR}` placeholder filled in.

```bash
grep -l '\${' .lithops/land-research/*.yaml && echo "BAD" || echo "OK"
```

---

## Step 5 — Gate A: local container parity (before any ECR push)

Builds the predict Docker image locally and runs it against a golden tile i fyou have that set up,
comparing the output to the reference TIF. Catches dependency-resolution,
TF-import, and graph-loading bugs in a few minutes without touching AWS.

```bash
make -C infra gate-a
```

Under the hood: `docker build` → mount `example/golden/`, `loaders/`, and
`models/` into the container → run `predict_tile_from_arrays` on tile
`1000X798Y` → compare to `example/golden/1000X798Y_FINAL.tif` using the
thresholds in `tests/parity/test_golden_parity.py` (baseline tier).

**Expected:** `[gate-a] PASSED`. If it fails, do not proceed to step 6 —
the image is broken and pushing it to ECR is wasted effort.

---

## Step 6 — Build and push runtime images

Lithops owns ECR repos and container images. Each runtime's repo lands in
the region its rendered config specifies (predict → us-east-1, S2/S1 →
us-west-2, DEM → eu-central-1).

```bash
AWS_PROFILE=<land_research_profile> make -C infra build-all ENV=land-research
```

Takes ~10–15 minutes the first time (TensorFlow, rasterio, etc.).

If Docker isn't running you'll see `Cannot connect to the Docker daemon` —
start Docker Desktop (or `colima start`) and rerun.

Need to force a rebuild after Dockerfile edits?

```bash
AWS_PROFILE=<land_research_profile> uv run lithops runtime build --no-cache \
    -f docker/PredictDockerfile -b aws_lambda \
    -c .lithops/land-research/config.predict.yaml ttc-predict-dev
```

**Expected:** four ECR repos populated, one image each. Verify with:

```bash
AWS_PROFILE=<land_research_profile> uv run lithops runtime list \
    -b aws_lambda -c .lithops/land-research/config.predict.yaml
```

Repeat with the other three configs to confirm all four runtimes register.

---

## Step 7 — Eagerly deploy the predict Lambda

Lithops deploys Lambdas lazily on first invocation, but eager deploy
surfaces IAM/role errors up front.

```bash
AWS_PROFILE=<land_research_profile> make -C infra deploy-predict ENV=land-research
```

**Expected:**

```bash
aws lambda get-function --function-name ttc-predict-dev \
    --region us-east-1 --profile <land_research_profile> \
    --query 'Code.ImageUri'
```

returns a us-east-1 ECR URI ending in `ttc-predict-dev:…`.

---

## Step 8 — Hello-world connectivity check

Confirms Lithops can reach Lambda at all before trying real inference.

```bash
AWS_PROFILE=<land_research_profile> uv run python -c "
import yaml, lithops
cfg = yaml.safe_load(open('.lithops/land-research/config.predict.yaml'))
fexec = lithops.FunctionExecutor(config=cfg)
f = fexec.call_async(lambda x: x * 2, (21,))
print('Result:', f.result(timeout=120))
"
```

**Expected:** `Result: 42`.

If this fails but step 7 succeeded, it's almost always one of:
(a) SSO token expired,
(b) the execution role's trust policy isn't assumable by Lambda (check
    step 2's output),
(c) the rendered config points at the wrong region (re-run step 4).

---

## Step 9 — Predict smoke test (single-tile round-trip)

First real inference. Invokes `ttc-predict-dev` on one golden tile that
already has ARD on `tof-output`, then validates the output TIF.

```bash
AWS_PROFILE=<land_research_profile> LITHOPS_ENV=land-research \
    uv run python scripts/predict_lambda_smoke.py
```

Default tile is `1000X871Y` year 2023. Pass `--tile 1000X798Y` for a
different one.

**Expected:** `SMOKE TEST PASSED` plus output TIF stats. First invocation
is a cold start (~90–120s image pull).

**If it fails with "access denied" on `tof-output`:** step 3's manual
runbook wasn't applied, or the policy doesn't reference the current role
ARN. Re-check both.

---

## Step 10 — Gate B: Lambda benchmark (recommended after image changes)

Quantifies p50/p95/p99 wallclock, cold-start behavior, throughput at
configured concurrency, and cross-region bytes. Useful to confirm the
us-east-1 move worked (cross-region bytes should be ≈ 0) and to establish
a baseline for future optimization.

```bash
AWS_PROFILE=<land_research_profile> LITHOPS_ENV=land-research \
    uv run python scripts/predict_lambda_benchmark.py --tiles 20
```

Writes `benchmarks/<UTC-date>-<git-sha>.csv` and prints a summary.

**Expected rough numbers** after a fresh deploy in us-east-1:

- p50 wallclock ≈ 180s
- cold-start batch median ≈ 200–250s (ECR image pull)
- warm-batch median ≈ 160–200s
- throughput ≈ 8–12 tiles/min at `max_workers=30`
- cross-region egress: **no (co-located)**

Cross-region bytes flipping to "YES" is a regression — double-check the
rendered predict config and ECR region.

---

## Step 11 — Use the pipeline

Export both env vars once per shell session:

```bash
export AWS_PROFILE=<land_research_profile>
export LITHOPS_ENV=land-research
```

Then drive the CLI as documented in
[`cli_workflows.md`](cli_workflows.md):

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
```

---

## Test gate summary

| Gate | When | What it confirms | Required? |
|---|---|---|---|
| Gate A (step 5) | Before every `build-all` / any image change | Container inference matches reference locally | Yes (cheap, fast) |
| Predict smoke (step 9) | After every deploy | End-to-end round-trip works over the network | Yes |
| Gate B — Lambda parity (step 10) | After every deploy | Deployed Lambda numerics match the golden references | **Yes — blocking** |
| Gate C — benchmark (step 11) | After any image or region change | Performance hasn't regressed; cross-region egress is 0 | Recommended |

Every one of these runs from the repo root, and every one of them is
idempotent — re-run them freely.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `The security token included in the request is expired` | SSO token timed out | `aws sso login --profile <land_research_profile>` (and `--profile <wri_acct_profile>` if it's also SSO) |
| `Cannot connect to the Docker daemon` during `build-all` or `gate-a` | Docker not running | Start Docker Desktop or `colima start` |
| ECR push fails with `no basic auth credentials` | Stale Docker auth token | `aws ecr get-login-password --region <runtime-region> --profile <land_research_profile> \| docker login --username AWS --password-stdin <account>.dkr.ecr.<runtime-region>.amazonaws.com` — region matches the runtime (predict = us-east-1; S2/S1 = us-west-2; DEM = eu-central-1) |
| Step 9 smoke fails with `ARD missing` | Golden ARD hasn't been uploaded for the selected tile | Pass `--tile 1000X798Y` (ARD always present) or run `gri-ttc download` for the tile first |
| Step 9 smoke fails with `AccessDenied` on `tof-output` | Step 3 manual runbook hasn't been applied, or the applied statements reference a stale role ARN | Re-run [`manual_wri_policy_update.md`](manual_wri_policy_update.md); confirm `lambda_role_arn` matches `aws lambda get-function-configuration --function-name ttc-predict-dev --query Role` |
| Lithops prints "runtime not deployed" on first invocation | Expected — Lithops deploys lazily if step 7 was skipped | Let it run once; subsequent invocations reuse the function |
| Step 10 Gate B parity is off by a few percent | Runtime numerical drift (TF/numpy version shift) | Diff the active image's pip list against a known-good one; rebuild if the drift is unexplained |
| Step 10 benchmark shows `cross-region egress: YES` | Rendered predict config still points at us-west-2 | Re-check `.lithops/land-research/config.predict.yaml` has `region: us-east-1` on all three fields; re-run step 4, then step 6 to rebuild in the right region |
