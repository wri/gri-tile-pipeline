# Release notes for gri-tile-pipeline

## 2026/09/16
1. Added control to pyproject.toml file for whether tests of golden datasets are run. Updated conftest with the logic for handling this control.
2. Modified download_golden to include additional sample files used by tests
3. Updated setup_uv_all.sh script to run download_golden script following deployment.

## 2026/06/29
1. Converted indicator-windows to use TreeCoverProjectPhaseYearRange enum definitions in gri-shared-library.
2. Modified missing.py to identify missing TTC over the inclusive ranges in TreeCoverProjectPhaseYearRange enum.
3. Optimized performance of missing.py as single-pass DuckDb queries.