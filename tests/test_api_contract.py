"""Snapshot of the HTTP contract, so a refactor cannot move it by accident.

Only about a third of the routes have behavioural tests, but every one of them
is a surface someone's browser, bookmark, or script depends on — 44 are written
down in the README. This file pins the shape of all of them at once: the set of
(method, path) pairs, each operation's parameters and status codes, and the
declared Pydantic models.

What it does NOT cover: most handlers return a bare ``dict`` with no
``response_model``, so their payload shape never reaches the OpenAPI document.
Those are guarded by the behavioural tests and by test_analysis_golden.py.
Treat a passing run here as "no route moved", not "no response changed".

Regenerate deliberately, never reflexively — a diff here means the public
surface changed, which is exactly the thing worth a second look:

    UPDATE_SNAPSHOTS=1 python -m pytest tests/test_api_contract.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# Must precede the api import: without it, importing the app opens (and creates)
# the developer's real database.
os.environ.setdefault("DB_PATH", ":memory:")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

_SNAPSHOT = Path(__file__).parent / "snapshots" / "api_contract.json"
_UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

_HTTP_METHODS = {"get", "put", "post", "delete", "patch", "head", "options", "trace"}

client = TestClient(app)


def _schema_ref(node: Any) -> str | None:
    """Return the component name a schema node points at, if any."""
    if not isinstance(node, dict):
        return None
    ref = node.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    for key in ("items", "additionalProperties"):
        inner = _schema_ref(node.get(key))
        if inner:
            return inner
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(key) or []:
            inner = _schema_ref(sub)
            if inner:
                return inner
    return None


def _operation_contract(op: dict[str, Any]) -> dict[str, Any]:
    params = sorted(
        f"{p.get('in')}:{p.get('name')}{'!' if p.get('required') else ''}"
        for p in op.get("parameters", [])
    )

    body = op.get("requestBody") or {}
    body_content = sorted((body.get("content") or {}).keys())

    responses: dict[str, Any] = {}
    for code, resp in sorted((op.get("responses") or {}).items()):
        content = (resp.get("content") or {}).get("application/json") or {}
        responses[code] = _schema_ref(content.get("schema"))

    return {
        "params": params,
        "body_content": body_content,
        "body_required": bool(body.get("required", False)),
        "body_model": _schema_ref(
            ((body.get("content") or {}).get("application/json") or {}).get("schema")
        ),
        "responses": responses,
    }


def _build_contract() -> dict[str, Any]:
    """Normalise the OpenAPI document down to the parts that are a promise.

    Deliberately drops summaries, descriptions, operationIds and titles: those
    are prose, and renaming a handler function should not fail this test.
    """
    spec = app.openapi()

    routes: dict[str, Any] = {}
    for path in sorted(spec.get("paths", {})):
        for method, op in sorted(spec["paths"][path].items()):
            if method.lower() not in _HTTP_METHODS:
                continue
            routes[f"{method.upper()} {path}"] = _operation_contract(op)

    models: dict[str, Any] = {}
    for name, schema in sorted((spec.get("components", {}).get("schemas", {})).items()):
        props = schema.get("properties") or {}
        models[name] = {
            "required": sorted(schema.get("required") or []),
            "properties": sorted(props.keys()),
        }

    return {"routes": routes, "models": models}


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    built = _build_contract()
    if _UPDATE or not _SNAPSHOT.exists():
        _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return built


@pytest.fixture(scope="module")
def stored() -> dict[str, Any]:
    if not _SNAPSHOT.exists():
        pytest.skip("snapshot not yet generated")
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def test_route_set_unchanged(contract, stored):
    """No route added, removed, or renamed."""
    live = set(contract["routes"])
    saved = set(stored["routes"])

    removed = sorted(saved - live)
    added = sorted(live - saved)
    assert not removed, (
        f"{len(removed)} route(s) disappeared — this breaks existing clients:\n  "
        + "\n  ".join(removed)
    )
    assert not added, (
        f"{len(added)} new route(s) not in the snapshot. If intended, regenerate with "
        f"UPDATE_SNAPSHOTS=1:\n  " + "\n  ".join(added)
    )


@pytest.mark.parametrize("route", sorted(_build_contract()["routes"]))
def test_route_contract_unchanged(route, contract, stored):
    """Each route keeps its parameters, request body, and status codes."""
    if route not in stored["routes"]:
        pytest.skip("new route; covered by test_route_set_unchanged")
    assert contract["routes"][route] == stored["routes"][route], (
        f"contract drift on {route}"
    )


def test_models_unchanged(contract, stored):
    """Pydantic request/response models keep their fields and required-ness."""
    assert contract["models"] == stored["models"]


# ---------------------------------------------------------------------------
# Surfaces that never reach the OpenAPI document
# ---------------------------------------------------------------------------


def test_root_redirects_to_breakdown():
    """`/` is where bookmarks point; it must keep landing on the same page."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
    assert resp.headers["location"] == "/frontend/breakdown.html"


def test_frontend_is_mounted():
    resp = client.get("/frontend/breakdown.html")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "path", ["/frontend/js/breakdown.js", "/frontend/theme.js", "/frontend/charts.js"]
)
def test_scripts_are_served_as_javascript(path):
    """A module script is refused outright if the type is not JavaScript.

    Python's mimetypes consults the Windows registry, where .js is commonly
    registered as text/plain. A classic <script> is served that way anyway; a
    <script type="module"> is not executed at all, so every page would load and
    do nothing — on that machine only, which is the worst kind of bug to be
    told about second-hand.
    """
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(("text/javascript", "application/javascript"))


@pytest.mark.parametrize("path", ["/frontend/breakdown.html", "/frontend/theme.js", "/frontend/theme.css"])
def test_frontend_assets_are_not_cached(path):
    """Documented behaviour with a real failure mode.

    api.py explains it: a page served fresh against a stale theme.js takes the
    section down with a ReferenceError. Losing this header reintroduces that.
    """
    resp = client.get(path)
    assert resp.status_code == 200
    assert "no-store" in resp.headers.get("cache-control", "")


def test_shared_scripts_in_subdirectories_are_served():
    """The mount has to reach into frontend/js/, not just the top level.

    A page whose <script src> 404s loads and then silently does nothing when a
    button is pressed, which is the same failure mode as a missing global.
    """
    resp = client.get("/frontend/js/steam-panel.js")
    assert resp.status_code == 200
    assert "no-store" in resp.headers.get("cache-control", "")


def test_images_stay_cacheable():
    """The counterpart: images are content-addressed and must NOT be no-store."""
    resp = client.get("/frontend/img/favicon.ico")
    if resp.status_code == 404:
        pytest.skip("favicon not present")
    assert "no-store" not in resp.headers.get("cache-control", "")


@pytest.mark.parametrize("literal,shadowing", [
    ("/api/matches/career-averages", "/api/matches/{match_id}"),
    ("/api/analyzer/version", None),
])
def test_literal_paths_are_registered_before_the_patterns_that_would_shadow_them(
    literal, shadowing
):
    """Starlette matches in registration order.

    Once the routes were split across routers, /api/matches/career-averages
    could be swallowed by /api/matches/{match_id} — the handler would run with
    match_id="career-averages" and answer 404 for a page that works. Nothing
    else in the suite would notice, because both routes still exist.
    """
    order = [r.path for r in app.routes if getattr(r, "path", None)]
    assert literal in order, f"{literal} is not registered at all"
    if shadowing:
        assert shadowing in order
        assert order.index(literal) < order.index(shadowing), (
            f"{literal} must be registered before {shadowing}"
        )


@pytest.mark.parametrize("path", [
    "/api/matches/career-averages",
    "/api/analyzer/version",
    "/api/performance",
    "/api/trends",
])
def test_literal_routes_actually_answer(path):
    """The ordering check above proves registration order; this proves the
    request reaches the handler that owns it rather than a parameterised one."""
    resp = client.get(path)
    assert resp.status_code != 404, f"{path} was swallowed by another route"


def test_cors_is_restricted_to_loopback():
    """Widening this exposes match history to any site the user is browsing."""
    evil = client.get("/api/config", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in {k.lower() for k in evil.headers}

    local = client.get("/api/config", headers={"Origin": "http://localhost:3000"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:3000"
