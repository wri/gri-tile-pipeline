# CLI workflows

Three recipes covering the core operations. Each recipe shows both input
routes — querying the TerraMatch geoparquet by filter, or providing your own
polygon file. Architectural background lives in
[`system_overview.md`](system_overview.md); this page is for getting work
done.

All examples assume you've completed [`setup.md`](setup.md) and have both
env vars exported:

```bash
export AWS_PROFILE=<land_research_aws_account>
export LITHOPS_ENV=land-research
```

---

## The two input routes

### Route 1 — filter `tm.geoparquet`

The authoritative polygon store is
`s3://<bucket_and_prefix>/tm.geoparquet`. For testing purposes a local copy could hang out 
at `temp/tm.geoparquet`; point the CLI at either via `--geoparquet`

**Columns available for filtering** (verified against the file):

| Column          | Type              | Notes                                             |
| --------------- | ----------------- | ------------------------------------------------- |
| `poly_uuid`     | VARCHAR           | Stable per-polygon identifier                     |
| `project_id`    | VARCHAR           | Project UUID                                      |
| `short_name`    | VARCHAR           | Human-readable project code (e.g. `GHA_22_INEC`)  |
| `framework_key` | VARCHAR           | Cohort framework (e.g. `terrafund-landscapes`)    |
| `cohort`        | BLOB (list)       | Cohort membership list — match with `list_contains` |
| `plantstart`    | DATE              | Planting start date                               |
| `country`       | VARCHAR           |                                                   |
| `calc_area`     | DOUBLE            | Polygon area in hectares                          |
| `ttc`           | MAP(INT, INT)     | Year → tree cover percentage (if already computed)|

Every CLI command that reads the geoparquet accepts the same filter flag
set:

| Flag                        | Behaviour                                     |
| --------------------------- | --------------------------------------------- |
| `--poly-uuid <uuid>`        | Repeatable — exact match on `poly_uuid`       |
| `--project-id <id>`         | Repeatable — exact match on `project_id`      |
| `--short-name <code>`       | Repeatable — exact match on `short_name`      |
| `--framework-key <key>`     | Repeatable — exact match on `framework_key`   |
| `--cohort <value>`          | Repeatable — matches via `list_contains(cohort, …)` |
| `--where '<SQL>'`           | Raw SQL WHERE expression, table alias `p`     |

The `--where` clause is read-only: `;` and the keywords DROP/DELETE/
INSERT/UPDATE/ALTER/CREATE/TRUNCATE/GRANT/REVOKE/ATTACH/COPY/PRAGMA are
rejected. Everything else — functions like `YEAR()`, `list_contains()`,
subqueries, `IN`, `BETWEEN` — works.

Example `--where` clauses:

```sql
-- All polygons planted in 2023 in Ghana
--where "country='GHA' AND YEAR(plantstart)=2023"

-- Polygons missing TTC values for their plantstart year
--where "ttc IS NULL OR cardinality(ttc)=0"

-- Two projects by UUID, restricted to a cohort
--where "project_id IN ('p1','p2') AND list_contains(cohort, 'priority')"

-- All polygons larger than 10 ha in a given framework
--where "framework_key='terrafund-landscapes' AND calc_area > 10"
```

### Route 2 — arbitrary polygon file

For polygons that don't live in `tm.geoparquet` (one-off analyses, external
datasets, bespoke AOIs): pass a GeoJSON/GeoPackage/Shapefile/Parquet file
directly. The commands `resolve`, `check`, `run`, and `stats` all accept
a polygon path — `resolve` turns it into a canonical `tiles.csv`; the rest
take that CSV as input.

The file must carry valid geometry in EPSG:4326. If it has a `plantstart`
column, use `--year-from-plantstart` to derive the prediction year
per-polygon; otherwise pass `--year N`.

---

## Recipe A — "Do we already have predictions for these polygons?"

Three flavors depending on what kind of artifact you want.

### A.1 — Full 4-phase Markdown report (scope → coverage → availability → missing CSV)

Best first step when planning a new campaign; works only for
geoparquet-resolvable inputs.

```bash
# Single project
gri-ttc report --short-name GHA_22_INEC

# Multiple short names
gri-ttc report --short-name GHA_22_INEC --short-name RWA_23_AEE

# By cohort / framework
gri-ttc report --framework-key terrafund-landscapes

# Specific polygons
gri-ttc report --poly-uuid abc-123 --poly-uuid def-456

# Arbitrary SQL
gri-ttc report --where "country='GHA' AND YEAR(plantstart)=2023"

# Skip the S3 availability check (faster, no AWS needed)
gri-ttc report --short-name GHA_22_INEC --skip-s3
```

Artifacts land in `--output-dir` (default `.`):
- `ttc_status_<timestamp>.md` — the report
- `ttc_missing_tiles_<timestamp>.csv` — feed this to `gri-ttc run` to fill gaps

> `report` derives prediction year from `plantstart` (per-polygon). To
> check availability for a different year, use A.2 or A.3 below.

### A.2 — Availability-only, reusing the `run-project` pipeline (short_name)

`run-project --check-only` runs steps 1–4 (extract polygons → coverage →
identify tiles → check S3) and exits. Supports `--year` for overrides.

```bash
# Plantstart-derived year (default)
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --check-only --yes

# Force a specific prediction year
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --year 2025 --check-only --yes
```

Writes a missing-tiles CSV to `temp/<label>_missing_tiles.csv` (override
with `-o`). No markdown report, but the same 4 phases are logged to
stdout.

### A.3 — Availability-only for arbitrary polygon file

```bash
gri-ttc check polygons.geojson \
    --year 2023 \
    --dest s3://tof-output \
    --check-type predictions \
    -o missing.csv
```

If the polygon file has a `plantstart` column, use
`--year-from-plantstart` to derive year per-polygon instead of passing
`--year`.

`missing.csv` is a canonical tiles CSV (Year, X, Y, X_tile, Y_tile) — any
downstream command accepts it. Full walkthrough:
[guides/tiles_availability.md](guides/tiles_availability.md).

---

## Recipe B — "Generate ARD + prediction tiles for what's missing"

Takes a tiles CSV (from Recipe A) and runs the download+predict steps on
Lambda. Safe to re-run; `--skip-existing` avoids redoing work.

```bash
gri-ttc run missing.csv \
    --dest s3://tof-output \
    --steps download,predict \
    --skip-existing \
    --yes
```

**Preview before spending**

```bash
gri-ttc run missing.csv --dest s3://tof-output --dry-run
# Prints tile count, region breakdown, estimated Lambda cost
```

Standalone commands exist if you want to run just one stage:

```bash
gri-ttc download missing.csv --dest s3://tof-output --yes
gri-ttc predict missing.csv --dest s3://tof-output --yes
```

**Local dev-box mode** — useful when debugging:

```bash
gri-ttc predict missing.csv --dest ./local_tiles --local --max-workers 4
```

**Where outputs land**

| Artifact         | Path (under `--dest`)                                           |
| ---------------- | --------------------------------------------------------------- |
| Raw ARD (.hkl)   | `{year}/raw/{X_tile}/{Y_tile}/raw/{source}/…`                   |
| Predictions      | `{year}/tiles/{X_tile}/{Y_tile}/{X_tile}X{Y_tile}Y_FINAL.tif`   |

---

## Recipe C — "End-to-end: polygons in, per-polygon tree cover out"

The single-command orchestrator. Resolves polygons → checks what's
missing → downloads ARD → predicts → runs zonal statistics → writes
`results.csv`.

### Execution mode (read this first)

`run-project` fans both download and predict out to **AWS Lambda via
Lithops by default**. Pass `--local` to run them in-process on your
machine instead (useful for smoke tests without cloud). The chosen mode
is announced in the first log line:

```
Mode: LITHOPS / AWS Lambda (LITHOPS_ENV=land-research, predict runtime=ttc-predict-dev). ...
Mode: LOCAL (max_workers=1) — workers run in this process. ...
```

`LITHOPS_ENV` must be exported in Lambda mode (`LITHOPS_ENV=land-research`
in production; `datalab-test` for dev). If it's unset in Lambda mode the
predict step fails loud with a pointer to `make render`.

### Route 1 — geoparquet filter (short_name, SQL, etc.)

```bash
# Classic: a short name as positional arg
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes

# Stop after the 4 availability steps (no download/predict)
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --check-only --yes

# Force a specific prediction year
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --year 2025 --yes

# Multiple short names via flag
gri-ttc run-project \
    --short-name GHA_22_INEC --short-name RWA_23_AEE \
    --dest s3://tof-output --yes

# Carve by cohort
gri-ttc run-project --cohort priority --dest s3://tof-output --yes

# Specific polygons
gri-ttc run-project \
    --poly-uuid abc-123 --poly-uuid def-456 \
    --year 2023 --dest s3://tof-output --yes

# Arbitrary SQL
gri-ttc run-project \
    --where "framework_key='terrafund-landscapes' AND YEAR(plantstart)=2023" \
    --dest s3://tof-output --yes
```

### Route 2 — arbitrary polygon file

`run-project` itself only accepts geoparquet-resolvable inputs. For an
external polygon file (GeoJSON / GeoPackage / Shapefile / Parquet in
EPSG:4326), use the two-step `resolve` → `run` flow, which runs on
Lambda by default with a `--local` opt-in matching `run-project`:

```bash
# End-to-end (download + predict + stats) in Lambda mode
gri-ttc resolve my_polys.geojson --year 2023 -o tiles.csv
gri-ttc run tiles.csv \
    --dest s3://tof-output \
    --steps download,predict,stats \
    --polygons my_polys.geojson \
    --year 2023 \
    -o results.csv \
    --yes

# Just check tile availability (no compute) — mirrors run-project --check-only
gri-ttc check my_polys.geojson \
    --year 2023 \
    --dest s3://tof-output \
    --check-type predictions \
    -o missing.csv
```

If the file has a `plantstart` column you can swap `--year 2023` for
`--year-from-plantstart` in both `resolve` and `check`.

### Useful `run-project` flags

| Flag                     | Effect                                                     |
| ------------------------ | ---------------------------------------------------------- |
| `--check-only`           | Stop after step 4 (write missing tiles CSV, exit)          |
| `--missing-only`         | Process only polygons missing TTC for their plantstart year |
| `--year N`               | Override plantstart-derived prediction year                |
| `--local`                | Run download+predict in-process (default: Lambda)          |
| `--max-workers N`        | Parallelism for `--local` mode (ignored on Lambda)         |
| `--skip-existing`        | (default) Skip tiles already on `--dest`                   |
| `--lulc-raster s3://…`   | Enable LULC-based error propagation                        |
| `--shift-error`          | Enable shift-error calculation                             |
| `--dry-run`              | Print plan; no side effects                                |

**Output (`results.csv`)**

One row per polygon with columns:

- `poly_uuid`, `short_name`, `project_id`, `plantstart`, `pred_year`
- `TTC` — mean tree cover (0–100)
- `area_HA` — polygon area in hectares
- Error columns if enabled (`TTC_shift_error`, `TTC_lulc_error`)

---

## Recipe D — "Push TTC results back to TerraMatch"

Once `gri-ttc stats` (or `run-project`) has produced a `results.csv`,
`gri-ttc tm-patch` sends each row to TerraMatch as a polygon indicator
via `PATCH /sitePolygons`. Default is dry-run — `--apply` is required
to actually write.

```bash
# Dry-run against staging
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 --env staging

# Single-polygon smoke test, then full staging apply
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 --env staging --limit 1 --apply
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 --env staging --apply \
    -o patch_outcomes.csv

# Production
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 --env production --apply
```

Inline from `run-project`:

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes \
    --tm-patch --tm-patch-project-id <TM_PROJECT_ID> \
    --tm-patch-env staging --tm-patch-apply
```

**Credentials.** `tm-patch` reads the bearer token from (in order)
`--token` → `GRI_TM_TOKEN` env var → `terramatch.token` in
`secrets.yaml`. See [configuration.md](configuration.md).

Full guide: [guides/terramatch_patch.md](guides/terramatch_patch.md).

---

## Run history and retries

Every fan-out step archives a `JobTracker` under `runs/<run_id>/`:

```bash
gri-ttc runs list                # all past runs
gri-ttc runs show <run_id>       # summary + breakdown
gri-ttc runs failed <run_id> -o retry.csv
gri-ttc run retry.csv --dest s3://tof-output --steps predict --yes
```

See [`system_overview.md §6`](system_overview.md) for the tracker schema and
the `runs/<run_id>/` layout.

---

## Cheat sheet

| I want to…                                                  | Command                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| Run everything end-to-end for a project                     | `gri-ttc run-project X --dest s3://tof-output --yes`        |
| Run everything for an arbitrary polygon file                | `gri-ttc resolve f.geojson --year Y -o tiles.csv` → `gri-ttc run tiles.csv --steps download,predict,stats --polygons f.geojson --year Y --yes` |
| Availability check for a project (no compute)               | `gri-ttc run-project X --dest s3://tof-output --check-only --yes` |
| Availability check for an arbitrary polygon file            | `gri-ttc check f.geojson --year Y --dest s3://tof-output -o missing.csv` |
| 4-phase Markdown availability report for a project          | `gri-ttc report --short-name X`                             |
| Run everything for a custom SQL filter                      | `gri-ttc run-project --where "..." --dest s3://tof-output`  |
| Fill just the prediction tiles                              | `gri-ttc run missing.csv --steps download,predict`          |
| Compute stats from polygons against existing prediction tiles | `gri-ttc stats polys.geojson --dest s3://tof-output --year Y` |
| Dry-run cost estimate for a tile list                       | `gri-ttc cost tiles.csv --include-predict`                  |
| Retry failed tiles from a past run                          | `gri-ttc runs failed <id> -o retry.csv` then `gri-ttc run retry.csv ...` |
| Quickly sanity-check the Lambda is live                     | `uv run python scripts/predict_lambda_smoke.py`             |
| Patch TTC results back to TerraMatch                        | `gri-ttc tm-patch --results results.csv --project-id <id> --year Y --env staging --apply` |
| Verify the workstation is ready to run jobs                 | `gri-ttc doctor` (add `--check-tm` for the TerraMatch API)  |
| Run a project on my laptop instead of Lambda                | `gri-ttc run-project X --dest ./local_tiles --local --max-workers 4 --yes` |
