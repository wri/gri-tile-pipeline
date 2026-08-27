# Release notes for gri-tile-pipeline

## 2026/06/29
1. Converted indicator-windows to use TreeCoverProjectPhaseYearRange enum definitions in gri-shared-library.
2. Modified missing.py to identify missing TTC over the inclusive ranges in TreeCoverProjectPhaseYearRange enum.
3. Optimized performance of missing.py as single-pass DuckDb queries.