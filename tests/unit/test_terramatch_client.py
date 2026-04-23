"""Tests for the TerraMatch HTTP client.

Mocks ``requests.Session`` so tests never touch the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gri_tile_pipeline.terramatch.client import TMApiError, TMClient


def _mock_response(status_code: int, json_body: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = text or str(json_body or "")
    return resp


def test_list_polygons_single_page_no_cursor():
    session = MagicMock()
    session.get.return_value = _mock_response(200, {
        "data": [
            {"id": "poly-a", "attributes": {"siteId": "site-1"}},
            {"id": "poly-b", "attributes": {"siteId": "site-1"}},
        ],
    })
    client = TMClient("https://tm.example/api", "tok", session=session)
    items = list(client.list_site_polygons("proj-1"))
    assert [i["id"] for i in items] == ["poly-a", "poly-b"]
    assert session.get.call_count == 1


def test_list_polygons_pagination_stops_on_stale_cursor():
    """If cursor doesn't advance between pages, we must stop rather than loop forever."""
    session = MagicMock()
    page_1 = _mock_response(200, {
        "data": [
            {"id": "poly-a", "meta": {"page": {"cursor": "c1"}}},
            {"id": "poly-b", "meta": {"page": {"cursor": "c1"}}},
        ],
    })
    page_2 = _mock_response(200, {
        "data": [
            {"id": "poly-c", "meta": {"page": {"cursor": "c1"}}},  # same cursor
        ],
    })
    session.get.side_effect = [page_1, page_2]
    client = TMClient("https://tm.example/api", "tok", session=session)
    items = list(client.list_site_polygons("proj-1"))
    assert [i["id"] for i in items] == ["poly-a", "poly-b", "poly-c"]
    assert session.get.call_count == 2


def test_list_polygons_pagination_advances():
    session = MagicMock()
    page_1 = _mock_response(200, {
        "data": [
            {"id": "poly-a", "meta": {"page": {"cursor": "c1"}}},
            {"id": "poly-b", "meta": {"page": {"cursor": "c1"}}},
        ],
    })
    page_2 = _mock_response(200, {
        "data": [
            {"id": "poly-c", "meta": {"page": {"cursor": "c2"}}},
        ],
    })
    page_3 = _mock_response(200, {"data": []})
    session.get.side_effect = [page_1, page_2, page_3]
    client = TMClient("https://tm.example/api", "tok", session=session)
    items = list(client.list_site_polygons("proj-1", page_size=2))
    assert [i["id"] for i in items] == ["poly-a", "poly-b", "poly-c"]
    # The third GET param should include page[after]=c2
    _, third_call_kwargs = session.get.call_args_list[2]
    assert third_call_kwargs["params"]["page[after]"] == "c2"


def test_get_non_200_raises():
    session = MagicMock()
    session.get.return_value = _mock_response(403, text="forbidden")
    client = TMClient("https://tm.example/api", "bad", session=session)
    with pytest.raises(TMApiError) as exc:
        list(client.list_site_polygons("proj-1"))
    assert exc.value.status == 403
    assert "forbidden" in exc.value.body


def test_patch_sends_expected_payload():
    session = MagicMock()
    session.patch.return_value = _mock_response(200, {"data": []})
    client = TMClient("https://tm.example/api", "tok", session=session)
    indicators = [{"indicatorSlug": "treeCover", "yearOfAnalysis": 2023,
                   "projectPhase": "implementation", "percentCover": 85.5}]
    resp = client.patch_site_polygon("poly-a", indicators)
    assert resp.status_code == 200
    _, kwargs = session.patch.call_args
    payload = kwargs["json"]
    assert payload["data"][0]["type"] == "sitePolygons"
    assert payload["data"][0]["id"] == "poly-a"
    assert payload["data"][0]["attributes"]["indicators"] == indicators
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


def test_patch_non_2xx_raises():
    session = MagicMock()
    session.patch.return_value = _mock_response(500, text="boom")
    client = TMClient("https://tm.example/api", "tok", session=session)
    with pytest.raises(TMApiError) as exc:
        client.patch_site_polygon("poly-a", [{"indicatorSlug": "treeCover"}])
    assert exc.value.status == 500
