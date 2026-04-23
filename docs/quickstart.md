# Quickstart

From `git clone` to a working `results.csv` in about ten minutes,
assuming (a) Python 3.11+ is available, (b) you already have AWS
credentials with access to the `land-research` account, and (c) a
TerraMatch project has been picked out. Infra setup (Terraform,
Lithops runtime builds) is covered separately in
[setup.md](setup.md) — follow that first if this is a cold account.

---

## 1. Install

```bash
git clone <this-repo> gri-tile-pipeline
cd gri-tile-pipeline
uv sync --extra all
```

`--extra all` pulls in loaders, predict, and zonal-stats dependencies.
If you only need to orchestrate (not run predict locally), `uv sync`
alone is enough.

Confirm the CLI is on your PATH:

```bash
uv run gri-ttc --version
```

---

## 2. Create your config

```bash
cp config_template.yaml  config.yaml
cp secrets_template.yaml secrets.yaml     # gitignored — never commit
```

Edit `secrets.yaml` only if you plan to use `gri-ttc tm-patch`; paste
your TerraMatch bearer token under `terramatch.token`. You can skip
this for now and come back to it when you need the patch-back step.

Full schema reference: [configuration.md](configuration.md).

---

## 3. Verify your workstation

```bash
export AWS_PROFILE=<account_profile>           # or your equivalent
export LITHOPS_ENV=land-research        # your rendered Lithops env
gri-ttc doctor
```

A healthy workstation shows `[OK  ]` for every check:

```
  [OK  ] config           loaded config.yaml
  [OK  ] secrets          parsed secrets.yaml
  [OK  ] aws-credentials  resolved via profile=<account_profile>
  [OK  ] lithops-env      LITHOPS_ENV=land-research -> .lithops/land-research/* present
  [OK  ] geoparquet       local temp/tm.geoparquet (3d old)
```

Add `--check-tm` to additionally ping the staging TerraMatch API with
your token. Fix any `[FAIL]` (each line has a `hint:`) before
spending compute.

---

## 4. Run your first project

Pick a TerraMatch short_name you know about (e.g. `GHA_22_INEC`).

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
```

That single command:

1. Queries `tm.geoparquet` for the project's polygons.
2. Resolves them to tiles via the canonical tile grid.
3. Reports how many already have ARD / predictions on S3.
4. Downloads missing ARD (DEM + S1 + S2) via Lithops on Lambda.
5. Runs TF inference on the tiles that now have ARD.
6. Computes per-polygon zonal tree cover statistics.
7. Writes `results.csv` with one row per polygon.

> **Execution mode.** Download and predict run on **AWS Lambda** by
> default (`LITHOPS_ENV` must be exported — the first log line prints
> the resolved mode). Pass `--local` to run workers in-process on your
> machine instead; useful for smoke tests without cloud access.

Safer dry-run first:

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --dry-run
```

Just the "what's missing" check, no compute:

```bash
# Full 4-phase Markdown report (year derived from plantstart)
gri-ttc report --short-name GHA_22_INEC --skip-s3

# Availability-only via the run-project pipeline (supports --year override)
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --check-only --yes
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --year 2025 --check-only --yes
```

Running from an arbitrary polygon file (GeoJSON / GPKG / shapefile /
parquet in EPSG:4326):

```bash
# Availability check
gri-ttc check polygons.geojson --year 2023 --dest s3://tof-output -o missing.csv

# End-to-end
gri-ttc resolve polygons.geojson --year 2023 -o tiles.csv
gri-ttc run tiles.csv --dest s3://tof-output \
    --steps download,predict,stats --polygons polygons.geojson --year 2023 --yes
```

---

## 5. Inspect outputs

```bash
head results.csv
# poly_uuid,short_name,project_id,plantstart,pred_year,ttc,area_HA,...
```

Column reference: see [guides/stats_run.md](guides/stats_run.md).

---

## 6. Push results back to TerraMatch (optional)
> [!WARNING]
> This hasn't been tested against production yet, will need to confirm on some small tests before using


```bash
# Dry-run first — prints what would be sent:
gri-ttc tm-patch --results results.csv --project-id <TM_PROJECT_ID> \
    --year 2023 --env staging

# Verify the plan looks right, then:
gri-ttc tm-patch --results results.csv --project-id <TM_PROJECT_ID> \
    --year 2023 --env staging --apply
```

Full walk-through: [guides/terramatch_patch.md](guides/terramatch_patch.md).

---

## Where to go next

- **Explain the pipeline**: [system_overview.md](system_overview.md)
- **Learn every CLI command**: [cli_workflows.md](cli_workflows.md)
- **Availability-check recipe**: [guides/tiles_availability.md](guides/tiles_availability.md)
- **Stats recipe (geoparquet queries)**: [guides/stats_run.md](guides/stats_run.md)
- **Set up the infrastructure from scratch**: [setup.md](setup.md)
