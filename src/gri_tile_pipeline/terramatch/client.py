"""HTTP client for the TerraMatch research v3 API.

Handles:

- Bearer-token auth via ``Authorization`` header.
- Paginated GET of ``/sitePolygons`` (cursor pattern).
- PATCH of ``/sitePolygons`` with one polygon per request.

Deliberately minimal: no retries, no async. Callers layer policy on top.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from loguru import logger

DEFAULT_STAGING_URL = "https://api-staging.terramatch.org/research/v3"
DEFAULT_PRODUCTION_URL = "https://api.terramatch.org/research/v3"


class TMApiError(Exception):
    """Raised on a non-2xx TerraMatch API response."""

    def __init__(self, status: int, body: str, message: str = "") -> None:
        self.status = status
        self.body = body
        super().__init__(message or f"TerraMatch API error {status}: {body[:200]}")


class TMClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        session: Any = None,
        timeout: float = 30.0,
    ) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = session or requests.Session()
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {token}"}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        logger.debug(f"TM GET {url} params={params}")
        resp = self.session.get(
            url, headers=self._headers, params=params, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise TMApiError(resp.status_code, resp.text)
        return resp.json()

    def _patch(self, path: str, payload: dict) -> Any:
        url = f"{self.base_url}{path}"
        logger.debug(f"TM PATCH {url} payload={payload}")
        resp = self.session.patch(
            url,
            headers={**self._headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        return resp

    def list_site_polygons(
        self, project_id: str, page_size: int = 100,
    ) -> Iterator[dict]:
        """Yield every polygon dict returned by ``/sitePolygons`` for ``project_id``."""
        params: dict = {
            "projectId[]": project_id,
            "page[size]": page_size,
        }
        last_cursor = None
        while True:
            data = self._get("/sitePolygons", params=params)
            items = data.get("data") or []
            if not items:
                return
            for item in items:
                yield item
            new_cursor = (
                items[-1].get("meta", {}).get("page", {}).get("cursor")
            )
            if not new_cursor or new_cursor == last_cursor:
                return
            last_cursor = new_cursor
            params["page[after]"] = last_cursor

    def patch_site_polygon(
        self, polygon_id: str, indicators: list[dict],
    ) -> Any:
        """PATCH ``/sitePolygons`` for a single polygon.

        Raises :class:`TMApiError` on non-2xx.
        """
        payload = {
            "data": [
                {
                    "type": "sitePolygons",
                    "id": polygon_id,
                    "attributes": {"indicators": indicators},
                }
            ]
        }
        resp = self._patch("/sitePolygons", payload)
        if resp.status_code >= 400:
            raise TMApiError(resp.status_code, resp.text)
        return resp
