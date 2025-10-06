## GRI Tile Pipeline

Pipeline for generating ARD tiles for TTC. This repository currently covers the ARD generation steps (DEM, S1, S2) and orchestrates parallel execution on AWS Lambda via Lithops. Final model inference is intended to be integrated in a subsequent phase.

## File reference

- The `lithops_job_tracker.py` script is the entrypoint for the ARD generation job.
- The `loaders/*` modules are the Lambda entrypoints for DEM, S1, and S2 ARD generation.
- The `scripts/handle_inbound_request.py` script converts an inbound JSON request into a tiles CSV consumed by `lithops_job_tracker.py`.
- The `.lithops/*.yaml` files contain per-environment and per-region Lithops configuration.

## Prerequisites

- AWS account with credentials configured locally (e.g., via AWS CLI or environment variables).
- Access to S3 buckets where outputs will be written.
- Python (recommend 3.11) for local orchestration.
- `uv` for local environment management.

Optional (for building Lambda runtime):
- Docker installed and available on your PATH.


## Local setup (uv)

This local environment is used for orchestration and helper scripts. The Lambda execution environment is built separately via Lithops runtime builds (see next section).

```bash
# Create and activate a local environment
uv venv
source .venv/bin/activate

# Install the minimal set of tools used by orchestration scripts
# (adjust if your environment requires more)
uv pip install lithops pyyaml boto3 pandas
```


## AWS and Lithops configuration

We use two AWS regions to colocate compute with data sources:
- `eu-central-1`: DEM and S1 tasks
- `us-west-2`: S2 tasks

The repository includes three configuration files under `.lithops/`:
- `config.euc1.yaml`: Configuration for `eu-central-1`
- `config.usw2.yaml`: Configuration for `us-west-2`
- `config.local.yaml`: Optional local execution settings

Set the Lambda runtime name/tag to `ttc-loaders-dev` (customize if needed) and verify AWS credentials and permissions to invoke Lambda and access S3.


## Build the Lithops runtime (Docker → AWS Lambda)

Build a Lambda-compatible Lithops runtime with the Python dependencies defined in `docker/PipDockerfile`.

```bash
# Build for us-west-2 (S2)
lithops runtime build -f docker/PipDockerfile -b aws_lambda -c .lithops/config.usw2.yaml ttc-loaders-dev

# Build for eu-central-1 (DEM, S1)
lithops runtime build -f docker/PipDockerfile -b aws_lambda -c .lithops/config.euc1.yaml ttc-loaders-dev
```

If you update dependencies, rebuild the runtimes. Confirm the runtime name configured in both `.lithops` files matches `ttc-loaders-dev`.


## Generate the tiles CSV

Use `scripts/handle_inbound_request.py` to translate an inbound request JSON into a tiles CSV consumed by the job tracker. A common pattern is to save the output as something like `example_tiles_to_process.csv`.

The CSV produced for the job tracker must contain the following columns:
- `Year`, `X`, `Y`, `Y_tile`, `X_tile`

Example:

```bash
python scripts/handle_inbound_request.py input.json --parquet data/tiledb.parquet > example_tiles_to_process.csv
```

Notes:
- The `--parquet data/tiledb.parquet` argument points to the parquet table used to map requests to tiles.
- See `scripts/handle_inbound_request.py` for additional options and input schema expectations.


## Run ARD jobs on AWS Lambda

Use `lithops_job_tracker.py` to fan out DEM + S1 (in `eu-central-1`) and S2 (in `us-west-2`) jobs. The script prints a pre-flight cost estimate based on historical runtimes and will prompt for confirmation before submitting any jobs.

```bash
python lithops_job_tracker.py example_tiles_to_process.csv \
  --dest s3://dev-ttc-lithops-usw2 \
  --plot
```

Key flags:
- `--dest`: Required. S3 destination prefix for outputs (e.g., `s3://bucket/prefix`).
- `--mem`: Optional. Lambda memory in MB (default 4096). Affects cost and speed.
- `--retries`: Optional. Automatic retries per task (default 3).
- `--plot`: Optional. Saves runtime plots into `plots/` for each region.
- `--report-dir`: Optional. Directory for job reports (default `job_reports`).

What to expect:
- A pre-flight cost estimate is printed (using `$0.00001667` per GB-second, with average durations `S1=14s`, `S2=62s`, `DEM=9s` and your specified memory). This is a rough estimate based on trial runs.
- An interactive prompt asks to proceed: type `y` to continue.
- Jobs are submitted to Lambda and tracked. At the end, reports are saved and a summary is printed.

Outputs:
- `job_reports/` contains a JSON report, CSV summary, and a text report summarizing successes and failures.
- `plots/` includes per-region execution timelines if `--plot` is used.
- ARD outputs are written under your `--dest` prefix in S3.

Planned enhancement:
- Before creating tasks, we plan to add an file check to skip jobs whose outputs already exist at `--dest`. This will likely compute expected keys and use `head_object` (S3) or `os.path.exists` (local) to filter tiles, and have an overwrite flag.




## Tests

Minimal tests live in `tests/`. You can run them with your preferred test runner after setting up the local environment.



## TODOs

- [ ] Integrate the final inference step into this pipeline after ARD generation.
- [ ] Set up default bucket write desitnation once we confirm the process works well for folks.
- [ ] Add pre-submission file checks and overwrite flagto avoid reprocessing completed tiles.
- [ ] Expand validation and data quality reporting.