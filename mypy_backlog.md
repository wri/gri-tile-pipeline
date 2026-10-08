# mypy backlog — `gri-tile-pipeline_for_review` (type-hint-only tree)

Generated 2026-09-14. Real `mypy` 2.3.1 run (same environment/stubs used for the
original type-hint review: `types-requests`, `types-PyYAML`, `pandas-stubs`,
`obstore==0.11.1` pinned to match `uv.lock`) against the current contents of
`gri-tile-pipeline_for_review` (`src/`, `packages/gri-tile-loaders/src/`, `scripts/`).
No AWS or TerraMatch calls were made to produce this — static analysis only.

**Result: 82 errors in 25 files.** For comparison, the same check against the
pristine `gri-tile-pipeline_gdp_122` original finds 184 errors in 29 files.

None of these require touching the type-hint-only tree again on their own — every
one of them needs a behavioral decision (fix the caller, fix the callee, or accept
the risk) to resolve, which is exactly the "bug fix" territory that was
deliberately kept out of this tree. This list is meant as the backlog for that
follow-up pass.

---

## 1. Reintroduced by today's type-hint/bug-fix split

These are errors that were **absent** from `for_review` immediately before today's
revert (the bug-fix hunks happened to also satisfy mypy) and are now back because
the underlying bug-fix code was moved out, per your instruction to fully separate
the two buckets.

### `packages/gri-tile-loaders/src/gri_tile_loaders/download_dem.py`
- **Line 265** — `Argument "expansion" to "make_bbox" has incompatible type "float"; expected "int"`. The reverted code passes `expansion=expansion / 30` (a `float`); the bug-fix version cast it with `int(expansion / 30)`.
- **Line 272** — `Argument 1 to "ensure_local_dirs_for_key" has incompatible type "AzureStore | GCSStore | HTTPStore | S3Store | LocalStore | MemoryStore"; expected "LocalStore"`. The reverted code calls this unconditionally; the bug-fix version guarded it with `if isinstance(store, LocalStore):`.

### `scripts/run_missing_predictions.py` (fully reverted to original)
- **Lines 39, 50, 70** — `Name "S3Store" is not defined`. The original script references `S3Store` in a type position without importing it (also present, identically, in `gdp_122` and in the never-touched `scripts/tile_parity_report.py`).

---

## 2. Already known — `cli.py` argument-mismatch cluster

Flagged in the very first mypy pass on this codebase and explicitly left
unfixed then and now, since resolving them means deciding whether the caller or
the callee is wrong, not correcting a hint.

`src/gri_tile_pipeline/cli.py`:
- **267** — `polygon_ids` to `list_polygons_missing_ttc`: `list[str] | None` vs expected `list[UUID] | None`
- **359** — `polygon_ids` to `generate_missing_tiles`: same `str` vs `UUID` mismatch
- **1572** — arg 3 to `run_mosaic`: `int | None` vs expected `int`
- **1580** — arg 1 to `run_zonal_stats`: `str | None` vs expected `str`
- **1580** — arg 3 to `run_zonal_stats`: `int | None` vs expected `int`
- **1670** — arg 1 to `resolve_tm_creds`: `str` vs expected `Literal['staging', 'production']`
- **1928** — assignment: `str | None` into a `str`-typed variable
- **1945** — `geoparquet` to `run_project_pipeline`: `bool` vs expected `str`
- **1956–1960** (5 lines) — `list(...)` called on `list[str] | None` where `Iterable[str]` is expected
- **1980** — `project_id` to `_invoke_tm_patch`: `str | None` vs expected `str`
- **1982** — `year` to `_invoke_tm_patch`: `int | None` vs expected `int`

---

## 3. Newly visible because of (kept) type-hint improvements

These lines had **no** mypy complaint in the original `gdp_122` — not because the
code was correct, but because the surrounding functions were untyped, so mypy
treated everything as `Any` and had nothing to check. Once real annotations were
added (legitimately kept as type-hint content), mypy could finally see mismatches
that were always true at the type level. None of these are regressions from
today's work; they're latent bugs the type-hint pass makes visible.

### `src/gri_tile_pipeline/storage/obstore_utils.py`
- **Line 55** — `**dict[str, object]` unpacked into `S3Store(...)` doesn't match any of its 5 keyword-only parameter types (`str | None`, `S3Config | None`, `ClientConfig | None`, `RetryConfig | None`, `S3CredentialProvider | None`) — the kwargs dict is too loosely typed for a `**kwargs` call into a `TypedDict`-like constructor.
- **Line 70** — assigning a `str | None` expression into a `str`-typed variable.

### `src/gri_tile_pipeline/tracking/job_tracker.py`
- **Line 105** — assigning a `dict[str, dict[str, float]]` into a target typed `dict[str, dict[str, int]] | float | str | None`.

### `packages/gri-tile-loaders/src/gri_tile_loaders/download_s1_rtc.py`
- **Line 1117** — `np.pad(...)` call doesn't match any overload for the given array/pad-width/mode argument combination.

### `packages/gri-tile-loaders/src/gri_tile_loaders/download_s1.py`
- **Line 1803** — same `np.pad(...)` overload mismatch as above (parallel code path).

### `packages/gri-tile-loaders/src/gri_tile_loaders/download_s2.py`
- **Line 832** — arg 2 to `identify_clouds_stac`: `list[float]` vs expected `tuple[float, float, float, float]`.
- **Line 877** — arg 2 to `download_sentinel2_stac`: same `list` vs 4-tuple mismatch.

### `src/gri_tile_pipeline/zonal/mosaic.py`
- **Line 77** — assigning a `tuple[float, float, float, float]` into an `int`-typed variable.

### `src/gri_tile_pipeline/execution/local_executor.py`
- **Lines 41, 71** — `tile_info` argument to `JobResult`: a `dict` typed with `Literal[...]` keys doesn't satisfy the declared `dict[str, Any]` parameter (two call sites, same mismatch).

### `src/gri_tile_pipeline/terramatch/patch.py`
- **Line 70** — assigning an `int` into a variable typed `str | None`.
- **Line 79** — return value `tuple[list[dict[Hashable, Any]], list[str]]` doesn't match the declared `tuple[list[dict[str, Any]], list[str]]` (a pandas `.to_dict()` result has `Hashable` keys, not `str`).
- **Line 140** — `.upper()` called on a value typed `Any | None` without a None-check.

### `src/gri_tile_pipeline/reporting/status_report.py`
- **Line 567** — `<` comparison between `int` and `dict[int, dict[str, object]]`, and between `int` and `None` (same expression, union type).
- **Line 571** — arg 2 to `write_tiles_csv`: `dict[int, dict[str, object]] | int | None` vs expected `list[dict[str, Any]]`.

### `src/gri_tile_pipeline/preprocessing/cloud_removal.py`
- **Line 405** — assigning an `ndarray` into a variable typed `list[Any]`.
- **Line 513** — assigning a boolean-dtype `ndarray` into a variable typed as `float64`-dtype `ndarray`.
- **Line 884** — passing an `ndarray[signedinteger]` where `MutableSequence[Any]` is expected.
- **Line 892** — passing an `ndarray[Any]` where `MutableSequence[Any]` is expected (parallel call).

### `src/gri_tile_pipeline/reporting/preview.py`
- **Line 108** — `extent` argument to `Axes.imshow`: `list[Any]` vs expected `tuple[float, float, float, float] | None`.

### `src/gri_tile_pipeline/steps/project_e2e.py`
- **Line 499** — arg 1 to `_extract_project`: `str | None` vs expected `str`.

### `scripts/predict_lambda_smoke.py`
- **Line 141** — args 2 and 3 to `prediction_key`: `int | float` vs expected `int` (surfaced because `tile: dict[str, int | float]` was given real type parameters upstream in this same file).

---

## 4. Pre-existing, unrelated to any type-hint work

Identical errors (same file, same message) in files that were never touched by
either the type-hint pass or today's revert. Listed for completeness since they
showed up in this run of the type-hint-only tree, but they are exactly as present
in `gdp_122` and are out of scope for this engagement unless you want them picked
up separately.

- **`src/gri_tile_pipeline/inference/frozen_graph.py:108`** — assigning a `str` into a `Path`-typed variable.
- **`src/gri_tile_pipeline/preprocessing/temporal_resampling.py`** — lines 124, 125, 134 (×2), 137, 139, 142, 146, 149: a value typed `object` used with `len()` and indexing throughout one function (8 errors).
- **`src/gri_tile_pipeline/exit_codes.py`** — lines 13–20: all 8 `Enum` members carry an explicit type annotation, which mypy's enum spec forbids.
- **`src/gri_tile_pipeline/reporting/audit.py:133`** — `status_counts` to `DropReport`: `dict[Hashable, int]` vs expected `dict[str, int]` (same pandas `Hashable`-key pattern as `terramatch/patch.py` above).
- **`scripts/describe_assets.py:172`** — `main` is annotated as returning a value but only ever returns `None`.
- **`scripts/describe_asset_by_dim.py:28`** — `list.__setitem__` called with two `int` arguments (looks like a slice-assignment typo).
- **`scripts/show_asset_by_dim.py:177`** — same `list.__setitem__` issue as above.
- **`scripts/tile_parity_report.py:615`** — args 3 and 6 to `_build_report`: `object` vs expected `ndarray`/`dict` respectively.
- **`scripts/print_raw_file_stats.py`** — lines 174, 182–185, 187, 308, 399, 400, 474: a matplotlib `Axes` vs `list[Axes]` mixup, two `tight_layout(rect=...)` calls passing a `list` instead of a 4-tuple, two `Any | None` args passed where an `ndarray` is required, and one `list[tuple[...]]` shape mismatch (10 errors, all in one file).

---

## Methodology note

Every comparison above was done by re-running mypy against a byte-for-byte
`gdp_122` original with the identical invocation and dependency versions, then
diffing error messages (not line numbers, since added type hints shift line
numbers even where nothing else changed). This is what let category 1 vs. 3 vs. 4
be told apart with confidence rather than guessed at.
