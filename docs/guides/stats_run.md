# Guide: TTC stats from a geoparquet query

Running zonal tree-cover statistics for polygons selected from
`tm.geoparquet`. Two flavours:

- **All-in-one**: `gri-ttc run-project` — filters the geoparquet, runs
  everything missing, emits `results.csv`.
- **Assemble-your-own**: `gri-ttc resolve` → `gri-ttc run --steps stats`
  when you want finer control or are driving from a polygon file.

Prerequisites: `gri-ttc doctor` clean. For anything beyond `--dry-run`
you'll need `AWS_PROFILE` and `LITHOPS_ENV` exported.

---

## Path A — one command

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
```

That's it: polygons for the project are pulled from `tm.geoparquet`,
the year comes from each polygon's `plantstart - 1`, and the full
pipeline runs. On success you get `results.csv` next to your shell.

### Filter variations (all interchangeable)

```bash
# Several short names
gri-ttc run-project \
    --short-name GHA_22_INEC --short-name RWA_23_AEE \
    --dest s3://tof-output --yes

# By framework / cohort
gri-ttc run-project --framework-key terrafund-landscapes --dest s3://tof-output --yes
gri-ttc run-project --cohort priority --dest s3://tof-output --yes

# Specific polygons
gri-ttc run-project --poly-uuid <uuid1> --poly-uuid <uuid2> \
    --year 2023 --dest s3://tof-output --yes

# Arbitrary SQL on the geoparquet (alias: p)
gri-ttc run-project --where "country='GHA' AND YEAR(plantstart)=2023" \
    --dest s3://tof-output --yes
```

`--where` is read-only; DDL/DML and `;` are rejected. All of these
accept the same filter flags as
[`gri-ttc report`](../cli_workflows.md#recipe-a----do-we-already-have-predictions-for-these-polygons).

### Useful `run-project` flags

| Flag                     | Effect                                                           |
| ------------------------ | ---------------------------------------------------------------- |
| `--dry-run`              | Print the plan + cost, no side effects.                          |
| `--check-only`           | Stop after the availability check; write missing-tiles CSV.      |
| `--missing-only`         | Skip polygons that already have a TTC value for their year.      |
| `--year N`               | Override the plantstart-derived prediction year. When set, the auto-generated output filename gets a `_<year>` suffix (e.g. `TEST_01_GRI_stats_2025.csv`) so it does not collide with the plantstart-relative run (`TEST_01_GRI_stats.csv`). |
| `--skip-existing`        | (default on) Skip tiles already at `--dest`.                     |
| `--lulc-raster s3://…`   | Enable LULC-based error propagation.                             |
| `--shift-error` / `--no-shift-error` | 8-direction shift-error metric. **On by default**; pass `--no-shift-error` to disable. |
| `--tm-patch` + `--tm-patch-project-id <id>` | After stats, patch results back to TerraMatch (dry-run by default; add `--tm-patch-apply`). |

---

## Path B — assemble-your-own (polygon file + existing predictions)

Use this when predictions already exist and you just want stats, or
when your polygons aren't in the geoparquet.

```bash
# 1. Resolve polygons to tiles (only needed for download/predict)
gri-ttc resolve polygons.geojson --year 2023 -o tiles.csv

# 2. Compute zonal stats for polygons against existing predictions
gri-ttc stats polygons.geojson \
    --dest s3://tof-output \
    --year 2023 \
    -o results.csv
```

Relevant `stats` flags:

| Flag                     | Effect                                                             |
| ------------------------ | ------------------------------------------------------------------ |
| `--include-cols a,b,c`   | Forward additional polygon columns into `results.csv`.             |
| `--lulc-raster s3://…`   | Override `zonal.lulc_raster_path`.                                 |
| `--shift-error` / `--no-shift-error` | 8-direction shift-error metric. **On by default** (`zonal.shift_error_enabled: true`); pass `--no-shift-error` to disable. |
| `--lookup-parquet PATH`  | Override `zonal.lookup_parquet`.                                   |

---

## `results.csv` schema

One row per polygon. Typical columns:

| Column                  | Type   | Notes                                                |
| ----------------------- | ------ | ---------------------------------------------------- |
| `poly_uuid`             | str    | Stable polygon identifier (auto-filled if missing).  |
| `short_name`            | str    | Project short name (from the geoparquet).            |
| `project_id`            | str    | Project UUID.                                        |
| `plantstart`            | date   | Planting start.                                      |
| `pred_year`             | int    | Year of the prediction used.                         |
| `ttc`                   | float  | Mean tree cover (0-100) over the polygon.            |
| `area_HA`               | float  | Polygon area in hectares.                            |
| `ttc_shift_error`       | float  | Present by default; absent if `--no-shift-error`.    |
| `ttc_lulc_error`        | float  | Present when `--lulc-raster` is set.                 |

Additional columns land if you passed `--include-cols`.

---

## Patching results back to TerraMatch

Once you've reviewed `results.csv`, you can push the indicators back
onto the matching TerraMatch polygons:

```bash
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 \
    --env staging               # dry-run in staging
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> --year 2023 \
    --env staging --apply       # actually send
```

Full guide: [terramatch_patch.md](terramatch_patch.md).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `No tiles resolved from input` | Filter matches zero polygons | Sanity-check with `gri-ttc report --skip-s3` using the same filter |
| `No prediction tiles intersect the polygons` in stats | Year mismatch or missing predictions | Re-run `check` to confirm availability; try a different `--year` |
| `results.csv` missing some polygons | They were dropped during stats | `gri-ttc audit-drops --request request.csv --stats results.csv` classifies the cause |
| `ttc_shift_error` column absent | `--no-shift-error` was passed, or `zonal.shift_error_enabled: false` in `config.yaml` | Drop `--no-shift-error` and remove/flip the config override — it is on by default |
