# Guide: patching TTC results back to TerraMatch

> [!WARNING]
> This feature is still experimental, needs to be vetted before use

After `gri-ttc stats` produces a `results.csv`, `gri-ttc tm-patch`
sends each row back to the TerraMatch database as a polygon
**indicator** (typically `treeCover`). Default behaviour is dry-run;
you opt into real writes with `--apply`.

Contract:

- One `PATCH /sitePolygons` request per polygon. No batching.
- Polygons are matched by `poly_uuid` (our side) ↔ polygon `id` (TM
  side, surfaced as `poly_id` / `polyId` in the API response).
- Unmatched rows are reported, never patched.
- Dry-run is the default.

---

## One-time setup

### 1. Obtain a bearer token

Get an API token from the TerraMatch research admin — same flow as
the reference `terramatch-researcher-api` uses. Keep the token out of
git.

### 2. Drop it into `secrets.yaml`

```yaml
# secrets.yaml
terramatch:
  token: ey…<your-token>…
```

Or export it each session:

```bash
export GRI_TM_TOKEN=ey…<your-token>…
```

The `GRI_TM_TOKEN` env var beats `secrets.yaml`; a `--token` CLI flag
beats both. See [configuration.md](../configuration.md#precedence).

### 3. Verify connectivity

```bash
gri-ttc doctor --check-tm              # defaults to staging
gri-ttc doctor --check-tm --tm-env production
```

A `[FAIL] terramatch …` line with `401 unauth` means the token isn't
valid for the target environment.

---

## The patch-back loop

A safe pattern end-to-end:

### 1. Compute stats (produces `results.csv`)

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
# or
gri-ttc stats polygons.geojson --dest s3://tof-output --year 2023 -o results.csv
```

### 2. Dry-run the patch

```bash
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> \
    --year 2023 \
    --env staging
```

This fetches every polygon TerraMatch knows about for the project,
matches them against your CSV rows, and prints the payload it *would*
send. No changes are made. Watch the summary:

```
Summary: total=42 sent=0 dryrun=38 unmatched=4 error=0
```

`unmatched` rows are polygons in `results.csv` whose `poly_uuid` isn't
returned by `/sitePolygons` for that project. Investigate before
proceeding — they could indicate a project-id mismatch or a polygon
that was deleted on the TM side.

### 3. Smoke-test with `--limit`

```bash
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> \
    --year 2023 \
    --env staging \
    --limit 1 --apply
```

Patches exactly one polygon. Confirm it looks right in the TerraMatch
staging UI before going wider.

### 4. Full staging apply

```bash
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> \
    --year 2023 \
    --env staging --apply \
    -o patch_outcomes.csv
```

The outcomes CSV has one row per result with `status ∈
{sent, dryrun, unmatched, error}`, the HTTP status code, and the
error message if any.

### 5. Promote to production

```bash
gri-ttc tm-patch --results results.csv \
    --project-id <TM_PROJECT_ID> \
    --year 2023 \
    --env production --apply \
    -o patch_outcomes_prod.csv
```

With `--env production --apply`, the CLI prompts for confirmation
unless `--yes` is also set. The `GRI_TM_TOKEN` env var (or
`secrets.yaml`) must hold a production-scoped token.

---

## Doing it inline from `run-project`

For a hot loop where you want stats + patch in one command:

```bash
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes \
    --tm-patch --tm-patch-project-id <TM_PROJECT_ID> \
    --tm-patch-env staging               # dry-run
# then:
gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes \
    --tm-patch --tm-patch-project-id <TM_PROJECT_ID> \
    --tm-patch-env staging --tm-patch-apply
```

The inline patch reads `pred_year` per row from the run's
`results.csv`, so polygons with different prediction years are
handled correctly.

---

## Payload shape

What `tm-patch` sends per polygon:

```json
{
  "data": [{
    "type": "sitePolygons",
    "id": "<poly_uuid>",
    "attributes": {
      "indicators": [{
        "indicatorSlug": "treeCover",
        "yearOfAnalysis": 2023,
        "projectPhase": "implementation",
        "percentCover": 82.5,
        "plusMinusPercent": 2.1
      }]
    }
  }]
}
```

Field sources:

- `id` — the row's `poly_uuid`.
- `indicatorSlug` — `--indicator-slug` (default `treeCover`).
- `yearOfAnalysis` — `--year-column` value if present on the row
  (default column name `year`, which `run-project` sets to `pred_year`),
  else `--year`.
- `projectPhase` — `--project-phase` (default `implementation`).
- `percentCover` — `--percent-column` (default `ttc`).
- `plusMinusPercent` — auto-detected from a shift-error column
  (`ttc_shift_error`, `ttc_error`, `*_stderr`, `*_uncertainty`) or
  whichever `--uncertainty-column` you pass; omitted when absent.

---

## Exit codes

| Code | Meaning                                                                |
| ---- | ---------------------------------------------------------------------- |
| 0    | Every row sent successfully (or dry-run completed with no errors).     |
| 1    | `PARTIAL_FAILURE` — some rows errored; others were sent/dry-run.       |
| 2    | `TOTAL_FAILURE` — every attempted row errored.                         |
| 3    | `BAD_INPUT` — bad flags, missing creds, project list call failed.      |
| 6    | `NO_WORK` — `results.csv` had no rows.                                 |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `TerraMatch token not found` | No flag, env var, or `secrets.yaml` entry | Set `GRI_TM_TOKEN` or add `terramatch.token` to `secrets.yaml` |
| Every row is `unmatched` | Wrong `--project-id`, or token doesn't have access to that project | Re-check the TM project id; try the same project via `curl` with your token |
| Single-row patch returns 422 | Payload field mismatch (e.g., an unknown `indicatorSlug`) | Inspect the `message` column in the outcomes CSV; slugs are case-sensitive |
| `poly_uuid` in results.csv doesn't look like a TM uuid | Polygon came from an external source, not `tm.geoparquet` | These rows can't match; filter them out before `tm-patch` |
| Patch ran twice — did I double-write? | `PATCH` is idempotent per HTTP, and TM overwrites indicators rather than appending | No harm; confirm in the UI |
