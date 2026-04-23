# Guide: tile availability checks

Before spending Lambda compute, ask: *which tiles do we already have
predictions for, and which are missing?* Pick the path that matches
your input source and output needs.

Prerequisites: `gri-ttc doctor` clean, and `AWS_PROFILE` / `LITHOPS_ENV`
exported.

---

## Choosing a path

| Starting point | Want… | Use |
|---|---|---|
| Short name in `tm.geoparquet` | Quick missing-tiles CSV | **Route 1a** — `run-project --check-only` |
| Short name in `tm.geoparquet` | Markdown report + missing-tiles CSV | **Route 1b** — `gri-ttc report` |
| Short name in `tm.geoparquet` | Availability for a specific year override | **Route 1a** with `--year N` |
| Arbitrary polygon file (GeoJSON/GPKG/Shapefile/Parquet) | Missing-tiles CSV | **Route 2** — `gri-ttc check polygons.geojson` |
| Existing tiles CSV | Verify what's there | **Route 3** — `gri-ttc check tiles.csv` |

All paths produce the same canonical `missing.csv` schema
(`Year, X, Y, X_tile, Y_tile`), so downstream commands (`run`,
`download`, `predict`) accept any of them.

---

## Route 1a — `run-project --check-only` for a geoparquet filter

Runs the first four steps of `run-project` (extract polygons → TTC
coverage → identify tiles → check S3) and exits without downloading or
predicting. Simplest single-command path.

```bash
# Classic: short_name with plantstart-derived year
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --check-only --yes

# Override prediction year
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --year 2025 --check-only --yes

# Any filter flag works in place of the short_name positional:
gri-ttc run-project --framework-key terrafund-landscapes \
    --dest s3://tof-output --check-only --yes
gri-ttc run-project --poly-uuid abc-123 --poly-uuid def-456 \
    --year 2023 --dest s3://tof-output --check-only --yes
gri-ttc run-project --where "country='GHA' AND YEAR(plantstart)=2023" \
    --dest s3://tof-output --check-only --yes
```

Output: `temp/<label>_missing_tiles.csv` (override with `-o`). Same
canonical tiles-CSV schema the rest of the pipeline consumes.

---

## Route 1b — `gri-ttc report` (full Markdown report)

Produces a human-readable 4-phase report alongside the missing-tiles
CSV. Year is always plantstart-derived; use Route 1a with `--year` if
you need a specific year.

```bash
# Single project
gri-ttc report --short-name GHA_22_INEC

# Any filter (same flag set as run-project)
gri-ttc report --framework-key terrafund-landscapes
gri-ttc report --poly-uuid abc-123 --poly-uuid def-456
gri-ttc report --where "country='GHA' AND YEAR(plantstart)=2023"

# Skip the S3 availability pass (faster, no AWS needed)
gri-ttc report --short-name GHA_22_INEC --skip-s3
```

Artifacts in `--output-dir` (default `.`):
- `ttc_status_<timestamp>.md` — the report
- `ttc_missing_tiles_<timestamp>.csv` — feed to `gri-ttc run` to fill gaps

The report includes:
1. Request scope (projects + polygons in the filter)
2. TTC coverage (how many already have a TTC value)
3. Tile availability (S3 presence check)
4. Missing-tile CSV

---

## Route 2 — arbitrary polygon file

When your polygons don't live in `tm.geoparquet` (bespoke AOIs, external
datasets, one-off analyses). The file must be in EPSG:4326.

```bash
gri-ttc check polygons.geojson \
    --year 2023 \
    --dest s3://tof-output \
    --check-type predictions \
    -o missing.csv
```

The input is spatially joined against the tile grid in
`data/tiledb.parquet` (configurable via `--lookup-parquet`), then each
resulting tile is HEADed on S3.

If the polygon file has a `plantstart` column, derive year per-polygon:

```bash
gri-ttc check polygons.geojson \
    --year-from-plantstart \
    --dest s3://tof-output \
    -o missing.csv
```

Useful flags:

| Flag                    | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `--check-type raw_ard`  | Check ARD HKL presence instead of final predictions.                    |
| `--exit-on-missing`     | Exit code 10 (`TILES_MISSING`) when any tile is missing — useful in CI. |
| `--region us-east-1`    | Override bucket region (default: config `zonal.tile_region`).           |

---

## Route 3 — existing tiles CSV

If you already have a canonical tiles CSV (e.g. from `gri-ttc resolve`
or a previous run):

```bash
gri-ttc resolve --short-name GHA_22_INEC -o tiles.csv    # optional if you don't have one
gri-ttc tiles validate tiles.csv                          # sanity-check schema
gri-ttc check tiles.csv --dest s3://tof-output -o missing.csv
```

This is the most granular path — useful when you've already filtered,
chunked, or massaged the tile list by other means.

---

## Interpreting the output

```
2026-04-22 14:01:12 | INFO | 178 existing, 12 missing out of 190 tiles
2026-04-22 14:01:12 | INFO | Missing tiles written to missing.csv
```

`missing.csv` columns: `Year, X, Y, X_tile, Y_tile`.

---

## Feed the missing list into compute

```bash
gri-ttc run missing.csv --dest s3://tof-output \
    --steps download,predict --yes
```

See [cli_workflows.md#recipe-a](../cli_workflows.md#recipe-a----do-we-already-have-predictions-for-these-polygons)
for fuller treatment of each path.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `check` finds nothing missing but predictions look wrong | Wrong `--check-type` (e.g. raw_ard when you meant predictions) | Re-run with `--check-type predictions` |
| `NoSuchBucket` / `AccessDenied` on `tof-output` | Cross-account policy not applied or expired SSO | See [manual_wri_policy_update.md](../manual_wri_policy_update.md) and re-run `aws sso login` |
| `No tiles resolved from input` | Empty filter — no polygons matched | Run the same filter under `gri-ttc report --skip-s3` to see the scope |
| Too many `missing.csv` rows on a known-good project | You're checking ARD when predictions already exist | `--check-type predictions` instead of `--check-type raw_ard` |
