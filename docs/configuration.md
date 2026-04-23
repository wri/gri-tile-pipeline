# Configuration

The pipeline reads three kinds of settings:

- **Runtime config** in `config.yaml` — non-secret settings (regions,
  Lambda memory, bucket names, Lithops paths).
- **Secrets** in `secrets.yaml` — the TerraMatch bearer token, plus
  any future credentials that should never land in git.
- **Environment variables and CLI flags** — per-invocation overrides.

Templates live at `config_template.yaml` and `secrets_template.yaml` in
the repo root. Copy them to `config.yaml` / `secrets.yaml` and edit.
`secrets.yaml` is gitignored.

---

## Precedence

Lowest to highest:

1. Built-in defaults in `src/gri_tile_pipeline/config.py`.
2. `LITHOPS_ENV` env var — retargets all four Lithops config paths to
   `.lithops/<env>/config.*.yaml`.
3. Keys in `config.yaml`.
4. Individual CLI flags (e.g. `--mem`, `--dest`, `--year`).

For the TerraMatch token, precedence is:

1. `--token` CLI flag (hidden; prefer env var).
2. `GRI_TM_TOKEN` environment variable.
3. `terramatch.token` in `secrets.yaml`.

The first non-empty source wins.

---

## `config.yaml`

### `pipeline`

| Key            | Type   | Default                | Purpose                                             |
| -------------- | ------ | ---------------------- | --------------------------------------------------- |
| `parquet_path` | string | `data/tiledb.parquet`  | Tile grid used by `resolve`, `stats`, `tiles.missing`. |

### `lithops`

Leave unset when `LITHOPS_ENV` is exported — the env var rewrites
these. Override individual keys if you need a custom path.

| Key              | Points at                                   |
| ---------------- | ------------------------------------------- |
| `euc1_config`    | DEM loader (`eu-central-1`)                 |
| `usw2_config`    | S2 loader (`us-west-2`)                     |
| `s1_usw2_config` | S1 RTC loader (`us-west-2`)                 |
| `predict_config` | TF predict (`us-east-1`, co-located with `tof-output`) |

### `download`

| Key         | Type | Default            | Purpose                        |
| ----------- | ---- | ------------------ | ------------------------------ |
| `runtime`   | str  | `ttc-loaders-dev`  | Lithops runtime tag (ECR image) |
| `memory_mb` | int  | `4096`             | Lambda memory; affects cost & speed |
| `retries`   | int  | `3`                | Worker retry limit             |

### `s1_rtc`  (legacy standalone S1 download)

| Key         | Type | Default       |
| ----------- | ---- | ------------- |
| `runtime`   | str  | `ttc-s1-dev`  |
| `memory_mb` | int  | `2048`        |
| `retries`   | int  | `3`           |

### `predict`

| Key            | Type | Default            | Purpose                          |
| -------------- | ---- | ------------------ | -------------------------------- |
| `runtime`      | str  | `ttc-predict-dev`  | Lithops runtime tag              |
| `memory_mb`    | int  | `6144`             | Lambda memory; sized from CloudWatch peak ~3437 MB + headroom. Verify against peak memory before dropping further. |
| `retries`      | int  | `2`                |                                  |
| `timeout_sec`  | int  | `600`              | Per-invocation cap               |
| `model_path`   | str  | `models`           | S3 URI or local path to `predict_graph-172.pb` |

### `zonal`

| Key                       | Type  | Default                 | Purpose                                                |
| ------------------------- | ----- | ----------------------- | ------------------------------------------------------ |
| `tile_bucket`             | str   | `tof-output`            | Bucket hosting prediction tiles                        |
| `tile_region`             | str   | `us-east-1`             | AWS region for `tile_bucket`                           |
| `small_sites_area_thresh` | float | `0.5`                   | Hectares cutoff for small-site handling                |
| `lulc_raster_path`        | str   | ""                      | LULC raster URI for error propagation (optional)       |
| `shift_error_enabled`     | bool  | `true`                  | Enable 8-direction shift-error metric (on by default)  |
| `lookup_parquet`          | str   | `data/tiledb.parquet`   | Tile grid lookup                                       |
| `lookup_csv`              | str   | ""                      | CSV fallback when parquet unavailable                  |

---

## `secrets.yaml`

### `terramatch`

| Key              | Type | Default                                             | Purpose                               |
| ---------------- | ---- | --------------------------------------------------- | ------------------------------------- |
| `token`          | str  | ""                                                  | Bearer token for the research API     |
| `staging_url`    | str  | `https://api-staging.terramatch.org/research/v3`    | Override the staging base URL         |
| `production_url` | str  | `https://api.terramatch.org/research/v3`            | Override the production base URL      |

---

## Environment variables

| Var               | Scope     | Effect                                                                                  |
| ----------------- | --------- | --------------------------------------------------------------------------------------- |
| `LITHOPS_ENV`     | Pipeline  | Retargets all Lithops config paths to `.lithops/<env>/config.*.yaml`.                   |
| `AWS_PROFILE`     | Pipeline  | Default AWS credentials profile (override per-run with `--aws-profile`).                |
| `AWS_REGION`      | Pipeline  | Default region for the AWS SDK.                                                         |
| `GRI_TM_TOKEN`    | tm-patch  | TerraMatch bearer token (beats `secrets.yaml`; loses to `--token`).                     |
| `GRI_GIT_SHA`     | Pipeline  | Override the provenance SHA stamped onto run outputs (useful in CI).                    |
| `DEST`            | download / predict / run | Default value for `--dest`.                                              |

---

## `LITHOPS_ENV` in practice

```bash
export LITHOPS_ENV=land-research
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
```

With `LITHOPS_ENV=land-research` set, the CLI reads:

- `.lithops/land-research/config.loaders-euc1.yaml`
- `.lithops/land-research/config.loaders-usw2.yaml`
- `.lithops/land-research/config.s1.yaml`
- `.lithops/land-research/config.predict.yaml`

These files are produced by `make -C infra render ENV=land-research`
from Terraform outputs — see [setup.md](setup.md) step 4.

---

## Verifying your config

```bash
gri-ttc --show-config           # print the resolved config as YAML
gri-ttc doctor                  # run every readiness check
gri-ttc doctor --check-tm       # also ping the TerraMatch API
```
