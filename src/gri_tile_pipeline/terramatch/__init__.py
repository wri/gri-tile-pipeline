"""TerraMatch API integration.

Primary entry points:

- :class:`TMClient` — HTTP client for the research v3 API.
- :func:`run_patch` — orchestration layer that patches TTC indicators
  onto polygons in a project.
- :func:`resolve_tm_creds` — resolves base URL + bearer token from
  CLI flags / env vars / ``secrets.yaml``.
"""

from gri_tile_pipeline.terramatch.client import (
    DEFAULT_PRODUCTION_URL,
    DEFAULT_STAGING_URL,
    TMApiError,
    TMClient,
)
from gri_tile_pipeline.terramatch.patch import (
    IndicatorSpec,
    PatchOutcome,
    build_indicator,
    build_poly_id_set,
    detect_uncertainty_column,
    load_results,
    run_patch,
    summarize,
)
from gri_tile_pipeline.terramatch.secrets import resolve_tm_creds

__all__ = [
    "DEFAULT_PRODUCTION_URL",
    "DEFAULT_STAGING_URL",
    "IndicatorSpec",
    "PatchOutcome",
    "TMApiError",
    "TMClient",
    "build_indicator",
    "build_poly_id_set",
    "detect_uncertainty_column",
    "load_results",
    "resolve_tm_creds",
    "run_patch",
    "summarize",
]
