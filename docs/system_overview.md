# Quick Ol' System Overview

This document describes how the pipeline is intended to operate end to end: the
problem it solves, the moving parts, how they're composed on the CLI, and how
the infrastructure is deployed and switched between accounts.

---

## 1. What the system does

Takes a description of "what we want tree-cover statistics for" — a
TerraMatch project, a set of polygons, or an explicit tile list — and produces
per-polygon tree-cover percentages with error propagation.

Four production pillars:

1. **Identify** which prediction tiles are needed.
2. **Generate ARD** (Analysis Ready Data) for the tiles that lack it: S2, S1
   RTC, DEM, and derived cloud masks.
3. **Infer** tree-cover predictions (`N X M Y_FINAL.tif`) from the ARD.
4. **Compute zonal statistics** for the polygons by mosaicking the relevant
   prediction tiles and running `exactextract` with error propagation.

Each pillar is idempotent and safe to re-run; outputs live in S3 with
deterministic keys, and the system preferentially skips what already exists.

---

## 2. Core concepts

- **Tile** — a 1/18° cell (~6.2 km at equator) in the canonical tile grid
  (`data/tiledb.parquet`). Identified by `(Year, X_tile, Y_tile)`.
- **Tiles CSV** — the canonical work unit. Columns `Year, X, Y, X_tile, Y_tile`.
  Every pillar accepts this shape.
- **ARD** — six HKL files per tile (DEM, S1, S2_10, S2_20, S2_dates, clouds)
  under `{year}/raw/{X}/{Y}/raw/…`.
- **Prediction tile** — a single GeoTIFF per `(year, X_tile, Y_tile)` at
  `{year}/tiles/{X}/{Y}/{X}X{Y}Y_FINAL.tif`, uint8 tree-cover percentage
  (nodata = 255).
- **Run** — one invocation of a step. Has a `run_id`; its `JobTracker` output
  is archived under `runs/<run_id>/`.

---

## 3. The CLI as a composition of small tools

Every operation is a subcommand of `gri-ttc`. Commands are intentionally small
and compose through the tiles CSV + `--dest` convention.

### Root-group flags (apply to every subcommand)

| Flag | Meaning |
|---|---|
| `--config PATH` | Pipeline YAML override |
| `-v` / `-vv` | DEBUG / TRACE logging |
| `-q` | WARNING-only logging |
| `--json` | Machine-readable JSON on stdout; text logs go to stderr |
| `--workers N` | Local-mode parallelism |
| `--dry-run` | Preview side effects only |
| `--yes` | Skip confirmation prompts |
| `--aws-profile NAME` | Override `AWS_PROFILE` for S3 |
| `--run-history-dir PATH` | Where runs get archived (default `runs/`) |
| `--run-id ID` | Override the auto-generated run id |

### Command tree

```
gri-ttc
├── resolve              Input → canonical tiles CSV
├── tiles
│   ├── missing          Tiles for polygons with null TTC
│   ├── split            Chunk a tiles CSV
│   └── validate         Schema (+ optional S3 presence) check
├── check                Availability on S3 (binary or --check-type)
├── cost                 Estimate Lambda cost without running
├── download             Fan out ARD generation via Lithops
├── download-s1-legacy   (to be folded into `download --s1-backend`)
├── predict              Fan out TF inference via Lithops
├── stats                Zonal statistics for polygons
├── run                  Orchestrator: --steps=download,predict,stats
├── run-project          TerraMatch-specific wrapper around `run`
├── report               4-phase TTC status report (markdown + CSV)
├── audit-drops          Classify polygons dropped during a run
├── preview-polygon      PNG of a polygon overlaid on its prediction tiles
└── runs
    ├── list             Past runs (run_id, step, counts)
    ├── show <id>        Full summary for one run
    ├── failed <id>      Emit a retryable failed-tiles CSV
    └── retry <id>       Suggest the retry command
```

### Unified input resolution

`gri-ttc resolve` turns **any** supported input shape into a canonical tiles
CSV:

- Existing tiles CSV → passthrough
- Request CSV (`project_id, plantstart_year`) → geoparquet join
- Polygon file (GeoJSON, GPKG, shp, parquet) → spatial join
- JSON request (legacy inbound) → DuckDB tile lookup
- Bare short name like `GHA_22_INEC` → query geoparquet

Downstream commands (`check`, `download`, `predict`, `stats`, `run`) accept
the canonical CSV. This is the single composition contract.

---

## 4. Data flow

```
                    ┌──────────────┐
     input shape ──►│    resolve   │──► tiles.csv  (year, X, Y, X_tile, Y_tile)
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    check     │──► missing.csv  (tiles that lack ARD or prediction on S3)
                    └──────────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
         ┌──────────────┐    ┌──────────────┐
         │   download   │    │   predict    │   ← Lithops fan-out on AWS Lambda
         │  (ARD, 3 srcs)│   │ (TF inference)│
         └──────┬───────┘    └──────┬────────┘
                │                   │
                ▼                   ▼
      s3://.../raw/...      s3://.../tiles/...
                                    │
                                    ▼
                            ┌──────────────┐
                            │    stats     │   ← polygon zonal stats + error propagation
                            └──────────────┘
                                    │
                                    ▼
                                results.csv
```

Every step writes to S3 at a deterministic key so the next step (or a
re-invocation) can decide what's already done.

---

## 5. Infrastructure model

### Ownership split

Terraform owns only what Lithops cannot create itself. Lithops keeps full
ownership of ECR, container images, and Lambda functions — deployed via
`lithops runtime build` / `lithops runtime deploy`.

| Resource | Owner |
|---|---|
| Lambda execution role | **Terraform** |
| Lithops state S3 buckets (per region) | **Terraform** |
| TF state backend (bucket + lock table) | **Terraform bootstrap** |
| Cross-account S3 access on `tof-output` | **Manual runbook** ([`manual_wri_policy_update.md`](manual_wri_policy_update.md)) — Terraform only emits the statement JSON |
| ECR repositories | **Lithops** (auto-created) |
| Container images (`ttc-loaders-dev`, `ttc-s1-dev`, `ttc-predict-dev`) | **Lithops** (`runtime build`) |
| Lambda functions | **Lithops** (`runtime deploy` or lazy) |
| Lithops configs (`.lithops/<env>/*.yaml`) | Rendered from Terraform outputs |

### Repository layout (`infra/`)

```
infra/
├── terraform/
│   ├── bootstrap/                     # TF state bucket + lock (one-shot)
│   ├── modules/
│   │   ├── lithops-iam-role/
│   │   ├── lithops-prereqs/
│   │   └── cross-account-s3-access/   # Output-only: emits statement JSON for the manual runbook
│   └── envs/land-research/            # Instantiates modules (land-research only — no wri provider)
├── lithops/
│   ├── config.*.yaml.tmpl             # Templates with ${VAR} placeholders
│   └── render.py                      # terraform output → envsubst → .lithops/<env>/
└── Makefile                           # render + lithops build + lithops deploy + gate-a
```

### Account model

- **land-research** — compute. ECR, Lambda, Lithops state buckets live here.
  Terraform manages this account.
- **wri** — data store. Owns `tof-output` (ARD + predictions) in us-east-1.
  Terraform **never** touches this account. The cross-account bucket policy
  on `tof-output` is applied manually, following
  [`manual_wri_policy_update.md`](manual_wri_policy_update.md).

### Per-runtime regions

Loaders co-locate with their data sources; predict co-locates with
`tof-output`:

| Runtime | Region | Why |
|---|---|---|
| `ttc-loaders-dev` (S2) | us-west-2 | Sentinel-2 Registry of Open Data |
| `ttc-loaders-dev` (DEM) | eu-central-1 | Copernicus DEM |
| `ttc-s1-dev` | us-west-2 | Azure (internet egress either way) |
| `ttc-predict-dev` | us-east-1 | `tof-output` is us-east-1 |

### Deployment flow

```
1. Terraform bootstrap (one-shot per account)
     cd infra/terraform/bootstrap && terraform apply
     → TF state bucket + DynamoDB lock

2. Terraform env apply (land-research only)
     cd infra/terraform/envs/land-research && terraform apply
     → IAM role, 3 Lithops state buckets (usw2/euc1/use1), cross-account
       statement JSON (output — not applied)

3. Manual wri policy update
     Follow docs/manual_wri_policy_update.md against the wri account.
     → cross-account grant on tof-output picked up on the next Lambda call

4. Render Lithops configs from Terraform outputs
     make -C infra render ENV=land-research
     → .lithops/land-research/config.{loaders-usw2,loaders-euc1,s1,predict}.yaml

5. Gate A — local container parity before any ECR push
     make -C infra gate-a
     → runs the predict image locally against a golden tile

6. Build and deploy Lithops runtimes
     make -C infra build-all ENV=land-research
     → Lithops pushes 4 images to ECR and (lazily) creates Lambdas

7. Gate B — Lambda parity
     PARITY_LAMBDA=1 uv run pytest tests/parity/test_lambda_parity.py -v -s
     → numeric parity vs golden reference TIFs; required before use

8. Switch the pipeline to the new account
     export AWS_PROFILE=resto-user LITHOPS_ENV=land-research
     gri-ttc run ...
```

### Config precedence

The CLI reads Lithops paths in this order (lowest → highest):

1. Built-in dataclass defaults
2. `LITHOPS_ENV` env var → `.lithops/<env>/config.*.yaml`
3. Per-key override in the pipeline YAML
4. Explicit `--config PATH` on the CLI

The `land-research` env takes over when `LITHOPS_ENV=land-research` is
exported; explicit config files always win if a user needs to pin something.

---

## 6. Failure handling and observability

### Run history

Every fan-out step records per-job outcomes in a `JobTracker`. On completion,
the tracker writes to `runs/<run_id>/`:

- `summary.json` — counts, timings, by-task-type breakdown
- `jobs.csv` — one row per Lithops job (tile, status, duration, error)
- `failed.csv` — tiles to retry (in canonical tiles-CSV shape)
- `report.md` — human-readable rollup

Those files are the single source of truth for "what happened." Query via:

```
gri-ttc runs list
gri-ttc runs show <run_id>
gri-ttc runs failed <run_id> -o retry.csv     # or pipe straight into `run`
gri-ttc runs retry <run_id>                   # prints the suggested command
```

### Granular availability

`tiles/availability.py` exposes two APIs:

- `check_availability(tiles, dest, check_type="raw_ard"|"predictions")` — binary
  per-tile existing/missing (today's default).
- `check_availability_by_source(tiles, dest, sources=...)` — per-source map
  `{(year, X_tile, Y_tile): {"s1": True, "s2_10": False, "dem": True, ...}}`.

Per-source lets downstream steps make smarter decisions (e.g., skip predict
for tiles missing S2, but still run tiles whose only gap is a cloud mask).

### Cost preflight

`gri-ttc cost tiles.csv --include-predict` estimates Lambda spend before
fan-out using `AVG_DURATIONS` and `PRICE_PER_GB_SEC`. Use before any run of
significant size.

### Retries and idempotence

- Lithops `RetryingFunctionExecutor` retries each worker invocation per the
  config (3 for ARD, 2 for predict).
- `--skip-existing` on `download` / `predict` / `run` avoids re-doing work
  already on S3.
- `gri-ttc runs failed <id>` extracts a clean retryable CSV — the standard
  recovery workflow.

---

## 7. Common operational workflows

### "Run a TerraMatch project end to end"

```
gri-ttc run-project GHA_22_INEC --dest s3://tof-output
```

Or step by step:

```
gri-ttc resolve GHA_22_INEC -o tiles.csv
gri-ttc check tiles.csv --dest s3://tof-output -o missing.csv
gri-ttc run missing.csv --dest s3://tof-output --steps download,predict,stats \
    --polygons polys.geojson --year 2023
```

### "Fill in TTC gaps for existing polygons"

```
gri-ttc tiles missing --short-name RWA_23_AEE -o gap.csv
gri-ttc run gap.csv --dest s3://tof-output 
```

### "What's the current TTC coverage for a cohort?"

```
gri-ttc report --framework-key terrafund-landscapes --skip-s3
```

Produces a 4-phase Markdown report (scope → coverage → tile availability →
missing-tiles CSV).

### "My run had failures — retry just those"

```
gri-ttc runs list
gri-ttc runs failed abc12345 -o retry.csv
gri-ttc run retry.csv --dest s3://tof-output --steps predict --yes
```

### "Preview a single polygon on its prediction tiles"

```
gri-ttc preview-polygon <poly_uuid> --year 2023 -o preview.png
```

### "Something got dropped — why?"

```
gri-ttc audit-drops --request request.csv --stats results.csv \
    -o dropped_report.csv
```

Classifies each dropped polygon: `invalid_but_repairable`, `zero_area`,
`wrong_geom_type`, `wkb_parse_error`, etc.

---

## 8. Intended interaction patterns

- **Humans** read reports (`gri-ttc report`, `runs show`), browse `runs/`,
  use `preview-polygon` for spot checks.
- **Scripts** use `--json` on every command to get a stable single-document
  output on stdout that jq / downstream tooling can consume; text logs stay on
  stderr.
- **Automation** pipes `resolve` → `check` → `run` → (on failure) `runs failed`
  | `run`. Every command is idempotent, every output is a canonical tiles
  CSV, and `--run-id` threads a stable identifier through the whole chain.

