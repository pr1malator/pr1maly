# Architecture

How the code is laid out, and why. The README describes what the app does; this
describes where things live and which decisions are load-bearing.

## The shape

```
api.py                    composition root — build the app, register routers (120 lines)
src/
  api/                    HTTP: routers, request/response models, shared state
    deps.py               the singletons and accessors every router builds on
    schemas.py            the Pydantic models
    routers/              one module per group of endpoints
  domain/                 pure logic — no HTTP, no database, no filesystem
    metrics/              the measurements, each registered with a version
    calibration/          numbers that came from outside the code
    callouts/             map zone data + the coordinate lookup
    blobs.py              reading the opaque JSON columns
  services/               the work the HTTP layer asks for, without the HTTP
  metrics/behavior.py     cross-match behavioural axes and role archetypes
  config/                 where files live, and how JSON config is read/written
  parser.py               .dem -> DataFrames (demoparser2)
  processor.py            composes the metrics into one stored match
  database.py             SQLite: schema, migrations, queries
  ai_service.py           LLM provider transport
frontend/                 five pages of markup
  js/                     one module per page, plus what they share
  theme.js  charts.js     classic scripts, shared with the landing site
  vendor/                 Tailwind and the fonts, so it runs offline
fetcher/                  Node companion that downloads demos from Valve
tools/build_release.py    builds the public source tree and landing site (dev only)
tools/build_demo.py       exports one match as the playable demo, anonymised (dev only)
tools/demo/demo-api.js    answers the demo's /api/ requests from files, in the browser
tools/domtest/            loads the pages in a DOM so the wiring can be tested (dev only)
docs/                     the landing site, and docs/demo/ — the app, served statically
tests/                    ~730 tests
```

## One process

The layers below are boundaries in the code, not processes on the machine.
`uvicorn api:app` is the whole application: the routers answer `/api/*`, and the
pages are a `StaticFiles` mount at `/frontend` on the same port, so `/` just
redirects into it. There is no build step, no dev server and no bundler: the
pages load ES modules the browser resolves itself, and Tailwind is a vendored
file. Docker runs one service.

Two things run alongside it inside that same process: Auto-Sync, a daemon
thread, and the Node fetcher — which is not a server either. `SteamJobRunner`
spawns `node fetch.js` when someone asks for a download, reads its stdout line
by line, and the process exits. One at a time, by design.

## How a request flows

A demo import, which is the path that touches the most:

```
POST /api/matches/upload
  src/api/routers/imports.py      validates, resolves whose match it is
  src/services/import_service.py  parse -> calculate -> stamp metadata
    src/parser.py                 demoparser2 -> ~20 DataFrames
    src/processor.py              builds enriched rounds, then runs the metrics
      src/domain/metrics/*        aim, utility, roles, impact, replay
      src/domain/calibration/*    win probability, HLTV coefficients
    src/database.py               one row in matches, one per round
```

A page load is shorter: `src/api/routers/analytics.py` reads rows through
`src/database.py`, decodes the JSON blobs through `src/domain/blobs.py`, and
aggregates with `src/metrics/behavior.py`.

## The boundaries that matter

**`api.py` builds the app and nothing else.** Endpoints live in
`src/api/routers/`. Anything shared between routers lives in `src/api/deps.py`;
nothing in `deps.py` imports a router, with one documented exception (the
Auto-Sync import step calls a route, resolved lazily).

**`src/domain/` does no I/O** beyond reading its own zone tables at import. It
takes the data it is given and returns values — no database handle, no request,
no subprocess. That is what makes a metric testable without a demo file.

**`src/services/` is where side effects live** — subprocesses, the filesystem,
the Steam fetcher, the background loop. Services raise domain exceptions
(`JobBusy`, `FetcherUnavailable`, `DemoFolderUnusable`); the HTTP layer turns
those into status codes. No service imports FastAPI.

**`src/database.py` owns all SQL.** There is none in `src/api/`. Batch readers
exist because the analytics pages walk every match: `/api/performance` over 200
matches is 3 queries, not 401.

## Decisions worth knowing

### Metrics are registered, versioned individually, and declare their inputs

`src/domain/metrics/registry.py`. Each measurement has an id, its own version,
and a `requires` set. Two things follow.

Only what changed goes stale. `ANALYZER_VERSION` is one number on every match,
so bumping it marks the whole library for re-parsing; a per-metric version marks
only the matches missing that measurement.

Some metrics can be recomputed with no demo. `requires` says what a metric
reads, and anything satisfied by `enriched_rounds` alone can be rebuilt straight
from SQLite, because those rounds are stored in `round_stats.enriched_json`.
That covers matches whose `.dem` the retention feature has already deleted.
`GET /api/metrics` reports which. There is a test proving a demo-free metric
gives identical output either way.

Versions are stored inside each metric's own blob, so there is no schema change.
`replay.frames` opts out — its keys are round numbers, and an extra one would
read as an extra round.

### Map zones are data, not code

`src/domain/callouts/zones/*.json` and `src/domain/metrics/role_zones/*.json`.
Order is load-bearing: zones are tested in sequence and the first match wins, so
a specific zone must precede the larger one enclosing it.

`tests/test_zone_data.py` cross-checks them — every role callout must name a zone
that exists, both bombsites must be claimed on both sides, no zone may be
shadowed. The failure mode without that is silent: a callout name matching
nothing contributes no score, so a role simply stops being detected.

**Known limitation:** on `de_nuke` the A and B bombsites have identical
rectangles, because the map is stacked and they share a 2D footprint. B Site
reads as A Site. Separating them needs a Z coordinate the zone model does not
carry. A test asserts the collision still exists, so the exception disappears
when someone fixes it properly.

### Config goes through one store

`src/config/store.py`. Reads merge declared defaults under whatever is on disk
and tolerate a byte-order mark, because these are files people open in editors.
Writes go via a temporary file and `os.replace` — `steam_sharecodes.json` holds
the match ledger and the Steam Web API key, and losing it to a crash mid-write
costs the user a credential they have to reissue. Files marked `private` are
chmod'ed before being moved into place, not after.

### Calibration is separated from the metrics that use it

`src/domain/calibration/`. Three artifacts with three provenances: the win
probability table is measured from a real corpus and regenerable, the HLTV
coefficients are published by a third party, the benchmark tiers are hand-set
and say so. Keeping them together makes "which of our numbers are measured and
which are guessed" answerable by looking in a directory.

The win-probability table has invariant tests, because it weights every impact
score in the app and a bad recalibration would otherwise change them silently.

### The registry versions measurements; the catalogue explains figures

Two tables, on purpose. `registry.py` holds the five units that get versioned
and recomputed — that is machinery. `catalogue.py` holds what each of the ~40
figures on screen measures, how it is derived, and where any tier shown against
it came from, which is writing.

Keeping them apart means the compute modules stay about computing, and the
wording has one home that the API, the interface and `METRICS.md` all read, so
those three cannot disagree. `tests/test_catalogue.py` checks both directions
against the golden analysis: nothing produced may go undescribed, and nothing
described may have stopped being produced.

The provenance field is the point. A tier shown to a player is either measured
from a corpus, published by someone else, or a line somebody drew — and until
this existed, the interface presented all three identically.

### The update check is the one outbound request, and it is opt-in

`src/services/updates.py`. The app promises no external services are required,
so the check is built so that promise stays literally true: off by default,
fetching a static file with nothing appended to it, comparing versions locally,
and never installing anything. Turning it off also discards what it learned.

The version it compares comes from `src/version.py`, which reads the installed
package metadata and falls back to `pyproject.toml` — the number used to be
written twice, and only one of them was checked against the git tag. What it
compares against is `docs/latest.json`, written by `tools/build_release.py` from
that same number, so a release cannot publish a version it did not build.

It is served from the project's own domain rather than a hosting service, with
`access_log off` on that one path. Every host of that file sees the same
unavoidable minimum — an IP, a timestamp, a user agent — so the choice was never
whether the request is visible but who keeps the record of it. Handing it to
someone else's pages service would have moved that record rather than deleted
it, and would have cost the one property worth having: that nobody, including
whoever publishes the file, can count who is running this.

The tests that matter most are the ones about silence:
`tests/test_updates.py` fails if anything reaches the network while the setting
is off, and `tests/test_release_build.py` fails if the shipped `data/` scaffold
arrives with it on.

### The demo is the app, not a picture of it

`docs/demo/` is the frontend — the same pages, the same modules, the same
drawing code — with one line added to each page. That line loads
`tools/demo/demo-api.js`, which replaces `window.fetch` so every `/api/` request
is answered from a file that `tools/build_demo.py` exported. There is no server
behind it and nothing to install to see it.

Two rules keep it honest.

It cannot become a fork. The copied files are byte-identical to `frontend/`,
`tools/build_release.py` refreshes them, and `tests/test_demo_site.py` fails on
any difference — including a page that is not exactly the app's page plus the
shim. The same test resolves every `src`, `href` and `import` in the copy,
because a missing ES module does not degrade: the whole import graph fails and
the page renders its empty state with a clean console, which is what shipping
without `reanalyze.js` looked like.

It cannot publish anybody. Match data is exported and then anonymised — Steam
IDs become ids that are not Steam IDs, players become stand-ins, match ids and
`.dem` filenames become fixed strings, free text is dropped rather than
scrubbed — and the export refuses to write if it can still find a real name or
id in its own output. Everything else is not exported at all: the account list,
AI configuration, and the Steam, storage and sync panels answer from constants
in `tools/build_demo.py`, because those are the endpoints that carry an API key
tail, a persona and four Steam IDs, and not reading them is a better guarantee
than scrubbing them. `tests/test_demo_site.py` then checks the published
directory from the outside, on every commit rather than only when someone
rebuilds.

### Markup names an action; JavaScript registers one

`frontend/js/actions.js`. A button says `data-action="openUploadModal"`, and the
file that defines that function registers it. Arguments travel in `data-args`
as JSON, several actions can be named in order, and every action is called as
`fn(...args, event, element)`.

This replaced 191 inline `onclick` attributes, and the reason is not style. An
inline handler is JavaScript evaluated against global scope, so every function
the markup called had to be a global — which is what made ES modules
impossible, and what made the wiring uncheckable: nothing but a browser could
tell you whether `onclick="doThing()"` still resolved. An action is a string,
so the names in the markup can be checked against the names in the registry
without running anything.

Listeners are attached to the elements themselves rather than delegated from
the document, because a delegated handler runs after the event has already
passed every ancestor — `stopPropagation` would no longer suppress a parent's
click, and one round card depends on exactly that. Elements rendered later are
picked up by a `MutationObserver`, so no render site has to remember.

### The pages are markup; the code is in frontend/js/

One ES module per page, importing what it needs. `theme.js` and `charts.js` are
the exception and stay classic scripts: `theme.js` has to run before the page
paints or a light-theme user sees a flash of the dark one, modules are deferred,
and the landing site under `docs/` loads both files the same way. The one thing
`theme.js` contributes to the markup is registered from the module side, in
`js/theme-actions.js`.

`js/api.js`, `js/escape.js` and `js/hooks.js` exist to make the graph acyclic.
The shared panels used to read `API` and `esc` out of whichever page had loaded
them, and call back into page functions that might not exist; the first two are
now imported, and `hooks` is where a page states the two things the panels need
back from it.

Tailwind, the icon font and the text fonts are vendored under `frontend/vendor/`
so the README's "no external services required" is true. The icon stylesheet
must keep its `.material-symbols-outlined` rule — icons are written as words and
become glyphs through an OpenType ligature, so losing that rule renders every
icon as its own name.

Scripts are served as `text/javascript` explicitly ([api.py](api.py)), because a
browser refuses to execute a module with any other type and Python's `mimetypes`
reads the Windows registry, where `.js` is often `text/plain`.

### The frontend is tested in a DOM, not by reading it

`tests/test_frontend_dom.py` loads every page in jsdom with its own scripts
running, replaces the page's functions with recorders, dispatches the event at
every element carrying a handler, and writes down what ran:

    element + event -> the functions it calls

That map is the contract. All 177 handlers across the five pages are in it, and
it is what made the frontend safe to restructure: the code moved out of the
pages into files, the handlers moved out of the markup into a registry, and the
files became modules — and the snapshot did not change through any of it.

jsdom does not implement module scripts, so `tools/domtest/harness.mjs` runs
them itself under Node with jsdom's globals installed, one process per page.
`tests/test_frontend_wiring.py` covers the static half: every action named in
markup is registered, no registration is dead, nothing is loaded over the
network, and no inline handler has crept back.

## Testing

The suite is the reason this layout could be arrived at from a 4,800-line
`api.py` without changing behaviour. Three snapshots carry most of that weight:

| Snapshot | Guards |
|---|---|
| `tests/snapshots/api_contract.json` | every route, model and parameter |
| `tests/snapshots/db_schema.json` | the DDL, and that an old database migrates forward |
| `tests/snapshots/analysis_golden.json` | the full analysis output, byte for byte |

Regenerate with `UPDATE_SNAPSHOTS=1 python -m pytest` — deliberately, after
reading the diff. A snapshot changing is the question, not the answer.

Route registration order is load-bearing and tested: Starlette matches in order,
so `/api/matches/career-averages` has to be registered before
`/api/matches/{match_id}` or the detail handler answers for it.

The boundaries above are assertions, not aspirations —
`tests/test_boundaries.py` fails if a service imports FastAPI, if anything in
`src/domain/` reaches for a database or a subprocess, or if SQL appears in
`src/api/`. And `tests/test_docs.py` fails if this file or the README names a
path that no longer exists, which is how a document starts being wrong.

## Compatibility

Existing databases work untouched — migrations are additive, and there is a test
that opens a pre-migration database and checks its rows survive. The JSON blob
columns are the only record of a match whose demo has been deleted, so
`src/domain/blobs.py` tolerates every historical shape rather than raising on
one.

`uvicorn api:app` still works, as do the Docker service name, ports and volume
paths.
