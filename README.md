# GRI Tile Pipeline

Tree tree-cover (TTC) statistics for restoration polygons, end to end.

Given a set of polygons — whether a TerraMatch project, a filter on
`tm.geoparquet`, or your own GeoJSON — this library resolves the tiles
they cover, downloads the ARD that's missing, runs TF inference on AWS
Lambda, computes per-polygon zonal statistics, and optionally patches the
results back to the TerraMatch database.

Everything is exposed as a single CLI: `gri-ttc`.

---

## At a glance

```
input (project / filter / polygon file)
  └─►  gri-ttc resolve      →  tiles.csv
       └─►  gri-ttc check   →  missing.csv
            └─►  gri-ttc run --steps download,predict,stats
                 └─►  results.csv
                      └─►  gri-ttc tm-patch  →  TerraMatch indicators
```

Every step is idempotent, writes to S3 at deterministic keys, and
`--skip-existing` avoids redoing work.

---

## Install

```bash
uv sync --extra all        # library + loaders + predict + zonal + dev
```

Python 3.11+ is required.

## Set up credentials

```bash
cp config_template.yaml  config.yaml
cp secrets_template.yaml secrets.yaml   # gitignored
# edit secrets.yaml: paste your TerraMatch bearer token (for tm-patch only)
```

## Verify the setup

```bash
gri-ttc doctor                    # checks config, AWS creds, Lithops env, geoparquet
gri-ttc doctor --check-tm         # also pings the staging TerraMatch API
```

Fix whatever `doctor` flags before running real jobs.

---

## Your first run

All four commands below default to running on AWS Lambda via Lithops —
export your creds first and the first log line confirms the mode:

```bash
export AWS_PROFILE=resto-user LITHOPS_ENV=land-research
```

Pass `--local` on any command to run workers in-process instead.

### Full end-to-end for a TerraMatch short_name

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
```

Resolves tiles → downloads ARD → predicts → computes polygon stats →
writes `results.csv`. Add `--tm-patch --tm-patch-project-id <TM_PROJ>`
to push outputs back to TerraMatch (dry-run by default; `--tm-patch-apply`
sends).

### Full end-to-end for an arbitrary polygon file

`run-project` is geoparquet-only. For an external polygon file
(GeoJSON / GPKG / Shapefile / Parquet in EPSG:4326), use `resolve` +
`run`:

```bash
gri-ttc resolve polygons.geojson --year 2023 -o tiles.csv
gri-ttc run tiles.csv --dest s3://tof-output \
    --steps download,predict,stats \
    --polygons polygons.geojson --year 2023 --yes
```

### Availability check for a short_name (no compute)

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --check-only --yes
```

Runs steps 1–4 (extract polygons → TTC coverage → identify tiles →
check S3) and writes `temp/<label>_missing_tiles.csv`. Supports
`--year N` to override the plantstart-derived year.

### Availability check for an arbitrary polygon file

```bash
gri-ttc check polygons.geojson --year 2023 \
    --dest s3://tof-output --check-type predictions -o missing.csv
```

See **[docs/quickstart.md](docs/quickstart.md)** for a fuller walk-through
and **[docs/cli_workflows.md](docs/cli_workflows.md)** for every flag.

---

## Common tasks

| I want to…                                               | Doc                                                        |
| -------------------------------------------------------- | ---------------------------------------------------------- |
| Install, verify, and run my first project                | [quickstart.md](docs/quickstart.md)                        |
| Understand the pipeline end to end                       | [system_overview.md](docs/system_overview.md)              |
| Stand up the AWS/Terraform/Lithops infrastructure        | [setup.md](docs/setup.md)                                  |
| See every CLI command with examples                      | [cli_workflows.md](docs/cli_workflows.md)                  |
| Configure `config.yaml` and `secrets.yaml`               | [configuration.md](docs/configuration.md)                  |
| **Check tile availability for a project**                | [guides/tiles_availability.md](docs/guides/tiles_availability.md) |
| **Compute TTC stats from a geoparquet query**            | [guides/stats_run.md](docs/guides/stats_run.md)            |
| **Patch TTC results back to TerraMatch**                 | [guides/terramatch_patch.md](docs/guides/terramatch_patch.md) |
| Manual cross-account `tof-output` bucket policy refresh  | [manual_wri_policy_update.md](docs/manual_wri_policy_update.md) |

---

## Infrastructure in one paragraph

Lithops fans out each pipeline step to AWS Lambda across three regions
(eu-central-1 for DEM+S1, us-west-2 for S2, us-east-1 for predict —
each co-located with its data source). Terraform provisions the IAM
role and per-region Lithops state buckets in the `land-research`
account; Lithops itself owns ECR repos and Lambda functions via
`runtime build` / `runtime deploy`. The `wri` account owns
`s3://tof-output` (where ARD and predictions live); a manual cross-
account bucket policy grants `land-research` Lambdas read/write access.
The full bring-up is in [docs/setup.md](docs/setup.md); the ownership
split is in [docs/system_overview.md §5](docs/system_overview.md).

---

## Tests

```bash
uv run pytest tests/unit/                                     # fast unit tests
uv run pytest tests/parity/test_golden_parity.py -v -s        # inference parity
```

---

## License

See [LICENSE](LICENSE).
