"""
pr1mealazyer — CS2 Local Analytics & Trend Tracker
FastAPI REST Backend

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

This module is the composition root and nothing else: it builds the app, sets
the middleware, mounts the frontend, and registers the routers. The endpoints
live in src/api/routers/, the state they share in src/api/deps.py, and the work
they do in src/services/.
"""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.deps import _resume_auto_sync
from src.api.routers import (
    ai,
    analytics,
    imports,
    matches,
    minimap,
    settings,
    steam,
    storage,
)
from src.config.settings import FRONTEND_DIR
from src.database import writability_problem
from src.version import APP_VERSION


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Say so at startup rather than letting the first import fail with SQLite's
    # "attempt to write a readonly database", which names neither the file nor
    # the reason.
    problem = writability_problem()
    if problem:
        print(f"WARNING: {problem}", flush=True)

    # Auto-Sync is meant to survive a restart: if it was left on, a docker
    # restart should not silently stop it.
    _resume_auto_sync()
    yield


app = FastAPI(
    title="pr1mealazyer API",
    description="CS2 demo analysis and match statistics",
    version=APP_VERSION,
    lifespan=_lifespan,
)

# The frontend is served from this same origin, so cross-origin access is only
# ever needed when someone runs the UI separately on another local port.
#
# It used to be allow_origins=["*"] with credentials, which Starlette turns into
# echoing whatever Origin asked. That let any website you happened to be
# visiting read this API while the app was running: your match history, the
# Steam IDs of everyone you have played with, your account list. Restricting it
# to loopback means the browser refuses to hand those responses to a page from
# anywhere else.
_LOCAL_ORIGIN_RE = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_LOCAL_ORIGIN_RE,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scripts and stylesheets are covered as well as pages. They used to be served
# with only an ETag, which leaves the browser free to apply heuristic freshness
# — so a page could load fresh HTML against a cached script from days earlier.
# That is not a cosmetic skew: the HTML calls into theme.js, and an older copy
# without the function it wants takes the whole section down with a
# ReferenceError. Images are left cacheable; they are content-addressed by name
# and only ever added.
_NO_CACHE_SUFFIXES = (".html", ".js", ".css")


@app.middleware("http")
async def no_cache_frontend(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(_NO_CACHE_SUFFIXES):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/health")
def health():
    """Is the app up?

    Deliberately cheap and deliberately not a database check. The container
    healthcheck calls this every thirty seconds; opening a connection each time
    would hold one open for the life of the container, and a long import would
    make the container report unhealthy while it was working perfectly.
    """
    return {"status": "ok", "version": app.version}


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/frontend/breakdown.html")


# Order matters. Starlette matches in registration order, so a literal path has
# to come before a parameterised one that would also match it — the upload
# routes in `imports` before /api/matches/{match_id} in `matches`, and
# /api/matches/career-averages before it too (source order does that one).
app.include_router(settings.router)
app.include_router(imports.router)
app.include_router(matches.router)
app.include_router(analytics.router)
app.include_router(ai.router)
app.include_router(steam.router)
app.include_router(storage.router)
app.include_router(minimap.router)

# Python's mimetypes reads the Windows registry, where .js is often registered
# as text/plain. A browser tolerates that for a classic <script>, but refuses
# to execute a module with a non-JavaScript type — the pages would load and do
# nothing at all, on that machine only. Stated here so it does not depend on
# how the host is configured.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

if FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
