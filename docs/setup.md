# Setup guide for standing up the pipeline in the `land-research` AWS account

Linear runbook for a first-time stand-up. Follow the steps in order; each
ends with a verification you should see pass before moving on.

**Single-account setup** — everything the pipeline needs lives in
`land-research` (`058755926933`). Terraform writes state to the shared
bucket `wri-restoration-terraform-state-lr` (used by other workflows in
the account too).

**Regions, by runtime:**

| Runtime | Region | Co-located with |
|---|---|---|
| S2 loader (`ttc-loaders-dev`) | us-west-2 | Sentinel-2 Registry of Open Data |
| DEM loader (`ttc-loaders-dev`) | eu-central-1 | Copernicus DEM |
| S1 RTC loader (`ttc-s1-dev`) | us-west-2 | Planetary Computer (Azure; internet egress either way) |
| Predict (`ttc-predict-dev`) | us-east-1 | `wri-restoration-geodata-ttc` |

Already set up and just want to drive the CLI? Skip to
[`cli_workflows.md`](cli_workflows.md).

---

## Prerequisites

**Local tools**

- Terraform ≥ 1.5
- AWS CLI v2
- Docker (daemon running — `docker info` must succeed)
- `uv` for Python deps

**Docker image store**

Lambda only accepts Docker manifest v2 schema 2 images. Docker Desktop's
containerd image store (Settings → General → "Use containerd for pulling
and storing images") produces OCI manifests, which Lambda rejects at
`CreateFunction` with `InvalidParameterValueException`. Uncheck that
setting before building runtimes. Confirm:

```bash
docker info --format '{{.Driver}}'   # should print overlay2, not overlayfs
```

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

Two permission sets on the `land-research` SSO account:

- `AWSAdministratorAccess-058755926933` — used for `terraform apply`
  (IAM role creation needs it) and for `lithops runtime build/deploy`
  against ECR.
- `LandResearchUser-058755926933` — day-to-day workflow runs (`gri-ttc`,
  `lithops runtime list`, reading S3).

The examples below use the admin profile because it has permission to
read the shared TF state key. Once the user role is granted state-read
permission, swap in `LandResearchUser-058755926933` for anything that
isn't Terraform.

---

## Step 0 — Verify credentials

```bash
aws sso login --profile AWSAdministratorAccess-058755926933
aws sts get-caller-identity --profile AWSAdministratorAccess-058755926933
```

SSO tokens expire (~8h). If a later step fails with `ExpiredToken`,
re-run `aws sso login`.

---

## Step 1 — Apply the land-research env

Creates the IAM role, three Lithops state buckets (one per region the
pipeline uses), and the TTC data bucket
`wri-restoration-geodata-ttc`.

```bash
cd infra/terraform/envs/land-research
cp terraform.tfvars.example terraform.tfvars
# (optional) edit terraform.tfvars to scope resource names

AWS_PROFILE=AWSAdministratorAccess-058755926933 terraform init \
    -backend-config=bucket=wri-restoration-terraform-state-lr \
    -backend-config=region=us-east-1 \
    -backend-config=dynamodb_table=terraform-state-lock
AWS_PROFILE=AWSAdministratorAccess-058755926933 terraform apply
```

**Expected:** apply succeeds and the key outputs are populated:

```bash
terraform -chdir=infra/terraform/envs/land-research output -raw lambda_role_arn
# arn:aws:iam::058755926933:role/lithops-execution-role

terraform -chdir=infra/terraform/envs/land-research output -raw state_bucket_use1
# wri-restoration-lithops-ttc-us-east-1  (or similar)
```

---

## Step 2 — Render Lithops configs

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra render ENV=land-research
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

## Step 3 — Gate A: local container parity (before any ECR push)

Builds the predict Docker image locally and runs it against a golden
tile, comparing the output to the reference TIF. Catches
dependency-resolution, TF-import, and graph-loading bugs in a few
minutes without touching AWS.

```bash
make -C infra gate-a
```

**Expected:** `[gate-a] PASSED`. If it fails, do not proceed to step 4 —
the image is broken and pushing it to ECR is wasted effort.

---

## Step 4 — Build and push runtime images

Lithops owns ECR repos and container images. Each runtime's repo lands
in the region its rendered config specifies (predict → us-east-1, S2/S1
→ us-west-2, DEM → eu-central-1).

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra build-all ENV=land-research
```

Takes ~10–15 minutes the first time (TensorFlow, rasterio, etc.).

If Docker isn't running you'll see `Cannot connect to the Docker daemon` —
start Docker Desktop (or `colima start`) and rerun.

**Expected:** four ECR repos populated, one image each. Verify with:

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 uv run lithops runtime list \
    -b aws_lambda -c .lithops/land-research/config.predict.yaml
```

Repeat with the other three configs to confirm all four runtimes register.

---

## Step 5 — Eagerly deploy the predict Lambda

Lithops deploys Lambdas lazily on first invocation, but eager deploy
surfaces IAM/role errors up front.

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 make -C infra deploy-predict ENV=land-research
```

**Expected:** Lithops creates a function named
`ttc-dev-lithops-worker-<hash>` in us-east-1. Find it and confirm the
image + role:

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 aws lambda list-functions \
    --region us-east-1 --query 'Functions[?contains(FunctionName, `ttc-dev-lithops-worker`)].FunctionName' \
    --output text

AWS_PROFILE=AWSAdministratorAccess-058755926933 aws lambda get-function \
    --function-name <function-name> --region us-east-1 \
    --query '{Role:Configuration.Role, Image:Code.ImageUri, State:Configuration.State}'
```

`Role` should be `arn:aws:iam::058755926933:role/lithops-execution-role`,
`Image` a us-east-1 ECR URI ending in `ttc-predict-dev:…`, and `State`
`Active`. If `State` is `Pending`, Lambda's still pulling — wait a
minute and re-check.

---

## Step 6 — Hello-world connectivity check

Confirms Lithops can reach Lambda at all before trying real inference.
This invocation deploys Lithops's generic `default-runtime-v312` layer
rather than the `ttc-predict-dev` image, so it only proves the
Lithops↔Lambda plumbing — not that predict itself works. The next step
exercises the real image.

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 uv run python -c "
import yaml, lithops
cfg = yaml.safe_load(open('.lithops/land-research/config.predict.yaml'))
fexec = lithops.FunctionExecutor(config=cfg)
fexec.call_async(lambda x: x * 2, 21)
print('Result:', fexec.get_result(timeout=180))
"
```

**Expected:** `Result: 42`. First call is a cold start (~90–120s image
pull).

If this fails but step 5 succeeded, it's almost always:
(a) SSO token expired,
(b) the execution role's trust policy isn't assumable by Lambda (check
    the IAM role trust policy),
(c) the rendered config points at the wrong region (re-run step 2).

---

## Step 7 — Predict smoke test (single-tile round-trip)

First real inference against the `ttc-predict-dev` image. Invokes it on
one golden tile that already has ARD on
`wri-restoration-geodata-ttc`, then validates the output TIF.

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 LITHOPS_ENV=land-research \
    uv run python scripts/predict_lambda_smoke.py
```

Default tile is `1000X871Y` year 2023.

**Expected:** `SMOKE TEST PASSED` plus output TIF stats. First
invocation is a cold start (~90–120s image pull).

**If it fails with `[PRECONDITION] ARD missing`:** the golden ARD
hasn't been generated on the new bucket yet. Generate it:

```bash
cat > /tmp/golden_tiles.csv <<'CSV'
Year,X,Y,X_tile,Y_tile
2023,-54.4722,-5.1389,1000,871
CSV

LITHOPS_ENV=land-research AWS_PROFILE=AWSAdministratorAccess-058755926933 \
    uv run gri-ttc download /tmp/golden_tiles.csv \
    --dest s3://wri-restoration-geodata-ttc --yes
```

Then re-run the smoke test. Four golden tiles exist in the smoke
test's `KNOWN_TILES`: 1000X871Y, 1000X798Y, 1000X799Y, 1000X800Y.

---

## Step 8 — Gate B: Lambda benchmark (recommended after image changes)

Quantifies p50/p95/p99 wallclock, cold-start behavior, throughput at
configured concurrency, and cross-region bytes. Confirms the us-east-1
move worked (cross-region bytes should be ≈ 0) and establishes a
baseline for future optimization. Requires all four `KNOWN_TILES` to
have ARD — see step 7 for how to generate them.

```bash
AWS_PROFILE=AWSAdministratorAccess-058755926933 LITHOPS_ENV=land-research \
    uv run python scripts/predict_lambda_benchmark.py --tiles 20
```

Writes `benchmarks/<UTC-date>-<git-sha>.csv` and prints a summary.

**Expected baseline** (2026-04 bring-up, memory=6144MB, max_workers=100):

- p50 wallclock ≈ 256s
- p95 wallclock ≈ 270s
- cold-start batch median ≈ 200–250s (ECR image pull)
- throughput ≈ 4–5 tiles/min at `max_workers=100`
- cross-region egress: **no (co-located)**

Cross-region bytes flipping to "YES" is a regression — double-check the
rendered predict config and ECR region.

---

## Step 9 — Use the pipeline

Export both env vars once per shell session:

```bash
export AWS_PROFILE=AWSAdministratorAccess-058755926933   # or LandResearchUser-058755926933 once granted state-read
export LITHOPS_ENV=land-research
```

Then drive the CLI as documented in
[`cli_workflows.md`](cli_workflows.md):

```bash
gri-ttc run-project GHA_22_INEC --dest s3://wri-restoration-geodata-ttc --yes
```

---

## Test gate summary

| Gate | When | What it confirms | Required? |
|---|---|---|---|
| Gate A (step 3) | Before every `build-all` / any image change | Container inference matches reference locally | Yes (cheap, fast) |
| Predict smoke (step 7) | After every deploy | End-to-end round-trip works over the network | Yes |
| Gate B — Lambda parity (step 8) | After every deploy | Deployed Lambda numerics match the golden references | **Yes — blocking** |

Every one of these runs from the repo root, and every one of them is
idempotent — re-run them freely.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `The security token included in the request is expired` | SSO token timed out | `aws sso login --profile AWSAdministratorAccess-058755926933` |
| `Cannot connect to the Docker daemon` during `build-all` or `gate-a` | Docker not running | Start Docker Desktop or `colima start` |
| `InvalidParameterValueException: image manifest ... not supported` at `CreateFunction` | Docker Desktop containerd image store produces OCI manifests, which Lambda rejects | Docker Desktop → Settings → General → uncheck "Use containerd for pulling and storing images" → Apply & Restart. Then `make -C infra build-all ENV=land-research` to rebuild. |
| ECR push fails with `no basic auth credentials` | Stale Docker auth token | `aws ecr get-login-password --region <runtime-region> --profile AWSAdministratorAccess-058755926933 \| docker login --username AWS --password-stdin 058755926933.dkr.ecr.<runtime-region>.amazonaws.com` — region matches the runtime (predict = us-east-1; S2/S1 = us-west-2; DEM = eu-central-1) |
| `terraform output` returns `403 Forbidden` on the state bucket | The caller's role lacks `s3:GetObject` on `wri-restoration-terraform-state-lr/gri-tile-pipeline/lr.tfstate` | Use `AWSAdministratorAccess-058755926933` (has read access) or grant `LandResearchUser-058755926933` read on that key |
| Step 7 smoke fails with `ARD missing` | Golden ARD hasn't been uploaded for the selected tile | Follow the download snippet in step 7, or pass a different `--tile` |
| DEM loader worker returns `None` silently with `rasterio.errors.RasterioIOError: AccessDenied` in CloudWatch | The Lambda execution role has no read permission on external public S3 buckets (Copernicus DEM, Earth Search) | Already granted in `lithops-iam-role`'s `ExternalPublicDataRead` statement. If a new loader hits a new bucket, add its ARN there. |
| S1 RTC loader worker returns `None` silently with `ModuleNotFoundError: planetary_computer` | The S1 image is missing the `planetary-computer` package | It's in `docker/PipDockerfile`. If the deployed Lambda is stale, delete the function and let Lithops redeploy with the current image. |
| Lithops prints "runtime not deployed" on first invocation | Expected — Lithops deploys lazily if step 5 was skipped | Let it run once; subsequent invocations reuse the function |
| Step 8 Gate B parity is off by a few percent | Runtime numerical drift (TF/numpy version shift) | Diff the active image's pip list against a known-good one; rebuild if the drift is unexplained |
| Step 8 benchmark shows `cross-region egress: YES` | Rendered predict config still points at us-west-2 | Re-check `.lithops/land-research/config.predict.yaml` has `region: us-east-1` on all three fields; re-run step 2, then step 4 to rebuild in the right region |
