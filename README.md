# pr1maly — CS2 Local Analytics & Trend Tracker

A standalone, locally-hosted application for Counter-Strike 2. Upload `.dem`
files, extract advanced player metrics, and track performance trends over
time — all stored locally with no external services required.

**[Try it without installing anything →](https://www.pr1maly.com/demo/match-breakdown.html?id=demo-01)**
A real 41-match history in the real interface, with every name and Steam ID
replaced. Click a round, open the minimap, play the 2D replay.

## Features

- **HLTV 2.0 Rating, ADR, KAST%, K/D, Impact, KPR/DPR** — ADR counts damage
  against the health the victim actually had, so overkill does not inflate it
- **Multi-kill rounds** (2K–5K), **clutch detection** (1vX), **trade detection**
- **Full 10-player scoreboard** with team split and rank extraction
- **Round-by-round enriched data**: economy, buy type, kill/death callout
  positions, opening duels, bomb events, utility usage
- **Aim & movement analysis**: aim score (0–100), counterstrafe detection,
  peek speed (how much speed was carried into the duel, and whether the
  counterstrafe held up at it), stance breakdown, per-weapon movement
  penalties, reaction time, crosshair placement quality, time-to-kill
  efficiency
- **Utility analysis**: grenades bought vs thrown vs wasted, flash efficiency
  (enemies flashed, blind duration, team flash penalty), HE/molotov damage per
  throw, smoke zone coverage, utility rating (0–100)
- **2D replay viewer**: canvas-based playback on radar images with all 10
  players, health bars, grenades (flash/HE/smoke/molotov with flight paths,
  durations, countdown timers), kill markers, and animated kill feed
- **Interactive minimap**: per-round kill/death/grenade positions on radar
- **5-axis behavioral chart**: Aggression, Trading, Isolation, Survival, Sniper
  — computed per side (CT/T) with win rate per dominant axis
- **Map-specific role detection**: positional roles classified from actual player
  coordinates with spider chart visualization
- **AI-powered coaching**: chat with OpenAI, Anthropic, Google Gemini, or
  Mistral about any match — round narratives with callout positions built
  automatically
- **AI role assessment**: AI analyzes your positioning patterns to identify CT &
  T roles per map
- **Performance trends**: HLTV Rating, ADR, KAST, K/D, Aim Score, Utility
  Rating over time with map filters
- **Multiple accounts**: track several Steam accounts, auto-detect player from
  `.dem.info` sidecars
- **Friends list**: friends highlighted on scoreboards
- **Sync folder**: point at your CS2 replays directory, scan for new demos per
  player, selectively import
- **Manual Fetch from Steam** *(optional)*: download demos directly from Valve
  instead of saving each one in-game — check for new matches and pull the
  backlog on demand. One-time setup lives in **Settings → Steam**. Requires the
  separate Node companion in `fetcher/`
- **Auto-Sync** *(optional)*: leave it on and it works through your backlog
  unattended — one demo downloaded, analysed, then a configurable pause
  (default 5 minutes) before the next. Pauses by itself while CS2 is running
- **Storage management**: a demo costs ~280 MB on disk but ~1.3 MB once
  analysed. Keep a rolling window of recent demos for re-analysis and reclaim
  the rest — **Settings → Storage**, with a live preview of what each retention
  number would free
- **Three upload methods**: single and bulk (one modal, two modes), plus folder sync
- **Map icons**: Valve's map badges in the Trend map picker, the match history
  list, and the match detail banner. Missing icons fall back to a text
  abbreviation, so an unrecognised map degrades rather than breaks
- **Context tagging**: annotate matches with notes and tags
- **Dark/light theme**: toggle on every page, respects system preference
- **Fully local**: SQLite storage, no cloud services (AI features need your own
  API key)

### Supported Maps

| Feature | Maps |
|---------|------|
| Full callouts + role detection | Mirage, Dust2, Inferno, Nuke, Ancient, Anubis, Overpass |
| Radar / minimap rendering | + Train, Vertigo, Office |

## Architecture

| Layer | Purpose | Where | Tech |
|-------|---------|-------|------|
| **1 – Parser** | Reads raw `.dem` files into DataFrames | `src/parser.py` | `demoparser2` |
| **2 – Metrics** | Filters events by Steam ID, calculates stats | `src/processor.py`, `src/domain/` | `pandas` |
| **3 – Storage** | Persists matches, round timelines, tags, players | `src/database.py` | `sqlite3` |
| **4 – Services** | Imports, Steam jobs, Auto-Sync, demo retention | `src/services/` | — |
| **5 – API** | REST endpoints + serves the frontend | `api.py`, `src/api/` | `FastAPI`, `uvicorn` |
| **6 – Frontend** | Interactive HTML/JS pages | `frontend/` | Vanilla JS, Chart.js |
| **7 – AI** | Match coaching & role assessment | `src/ai_service.py`, `src/services/ai_context.py` | OpenAI / Anthropic / Gemini / Mistral |

[ARCHITECTURE.md](ARCHITECTURE.md) covers this properly: how a request flows
through the layers, which boundaries are load-bearing and why, and the decisions
worth knowing before changing anything.

---

## Quick Start

### Prerequisites

- **Python 3.11+** (tested with 3.11 and 3.14)
- **pip** (comes with Python)
- **Docker** and **Docker Compose** *(only needed for Option B)*

### Option A — Run locally (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/pr1malator/pr1maly.git
cd pr1maly

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser. The app redirects you to the
Performance Breakdown page automatically.

### Option B — Run with Docker

```bash
# 1. Build and start
docker compose up --build

# 2. Open http://localhost:8000
# 3. Stop with:
docker compose down
```

The SQLite database is persisted in `./data/` on your host via a Docker volume.

> **Sync Folder**: your CS2 replays directory is mounted as `/demos` inside the
> container, which is the path you enter in the Sync Folder UI. The host side
> defaults to the standard Windows Steam location; on Linux or macOS, or if your
> library lives elsewhere, copy `.env.example` to `.env` and set `DEMO_HOST_DIR`.

The application runs as a non-root user (uid 1000) and the container reports a
health status, so `docker compose ps` tells you whether the app is actually
answering rather than merely running.

The container starts as root only long enough to hand `/app/data` over to that
user, then drops privileges. That step exists because an earlier version ran as
root, and a database it wrote stays root-owned — which the app can read and not
write, and which SQLite reports as *"attempt to write a readonly database"*
while naming neither the file nor the reason. If you are on an older image and
hit that, this fixes it without rebuilding:

```bash
docker compose exec -u root api chown -R 1000:1000 /app/data
```

To skip the root step altogether, run as yourself — the entrypoint notices and
does nothing:

```bash
docker compose run --user "$(id -u):$(id -g)" api
```

---

## First-Time Setup

1. **Open the app** at **http://localhost:8000**. You'll land on the
   Performance Breakdown page.

2. **Complete the onboarding wizard** — a guided 5-step modal appears on your
   first visit:
   - **Welcome** — quick intro to the app
   - **Features** — overview of what's available
   - **Identity Setup** — upload any `.dem` + `.dem.info` file, the app parses
     all 10 players. Click **"Set as Me"** on your name to create your account,
     and **"Add Friend"** on anyone you queue with. (Both steps are skippable
     and can be done later in Settings.)
   - **AI Config** *(optional)* — select a provider (OpenAI, Anthropic, Gemini,
     or Mistral), paste your API key, and pick a model. Skip this if you don't
     want AI features.
   - **Done** — you're ready to go.

3. **Upload demos** — three methods, all in the UI:
   - **Single upload**: click **Upload Demo** in the sidebar, pick your `.dem`
     file (and optionally a `.dem.info` for auto-dating), add notes/tags.
   - **Bulk upload**: same **Upload Demo** modal, switched to **Bulk** — select
     multiple `.dem` and `.dem.info` files at once.
   - **Sync folder**: click **Sync Folder** in the sidebar, point it at your
     CS2 replays directory (e.g.
     `C:\...\Counter-Strike Global Offensive\game\csgo\replays`), scan for new
     demos, and selectively import them.

4. **Browse your stats** — every page updates automatically once matches are
   uploaded.

> **Tip**: You can manage accounts, friends, AI config, and more anytime via
> the **Settings** button (gear icon) in the sidebar.

---

## Frontend Pages

The frontend is served at `/frontend/` and consists of five pages. Navigate
between them using the sidebar.

### Performance Dashboard (`performance.html`)

Career-level stats and trends overview.

- **KPI cards**: K/D Ratio, KAST%, HLTV Rating, Win Rate, Aim Score — each
  with a trend indicator comparing to your career average
- **Performance trend chart**: HLTV rating and ADR plotted over your last 20
  matches
- **Tactical AI feedback**: three-section AI analysis ("The Good", "The Bad",
  "The Ugly") highlighting strengths and areas for improvement
- **Recent matches table**: your latest matches with map, score, K/D/A, rating,
  and result — click any row to jump to the full match breakdown

### Performance Breakdown (`breakdown.html`)

Deep aggregated analytics across all your matches.

- **Overall stats**: HLTV Rating, ADR, K/D, KAST%, HS%, Win Rate, Aim Rating,
  Utility Rating with min/max range indicators
- **Map filter**: click any of the 9 competitive maps to filter all stats to
  that map
- **Mechanics card**: headshot %, opening duel K/D, top weapon breakdown
- **Side effectiveness**: CT vs T win rates, ADR, survival %, pistol round
  performance
- **Role detection**: heuristic + AI-powered role identification per side (e.g.
  "Entry Fragger", "B Anchor") with radar visualization of positioning patterns
- **5-axis behavioral chart**: Aggression, Trading, Isolation, Survival, Sniper
  scores per side

### Match Breakdown (`match-breakdown.html`)

Detailed analysis of a single match. Accessed by clicking a match from the
dashboard or by navigating to `match-breakdown.html?id=<match_id>`.

- **Match header**: map, score, result, HLTV Rating, ADR, Impact
- **Scoreboard**: your team vs enemy team with K/D/A, ADR, KAST, Rating
  (friends are highlighted)
- **Mechanics card**: HS%, K/D, KAST% with visual bars
- **Utility card**: enemies flashed, avg blind duration, HE damage, Molotov
  damage
- **Pattern recognition**: clutch win %, trade %, opening duels, multi-kill
  rounds
- **Side analysis**: CT vs T score, win rates, ADR, survival, pistol round
  badges
- **Aim analysis**: overall aim score (0–100), counterstrafe/movement analysis,
  peek speed with the counterstrafe rate split by how fast the peek was,
  stance breakdown (standing, counterstrafing, running), per-weapon movement
  penalty
- **AI match chat**: ask an LLM questions about the match — pre-built prompts
  for match overview, round-by-round analysis, economy, opening duels, and
  clutch/impact analysis
- **Interactive minimap**: per-round kill, death, and grenade positions
  overlaid on the radar image

### Callout Calibration (`calibrate.html`)

Developer tool for calibrating map coordinate → pixel position mappings.

- Select a map from the dropdown, then click on the radar image to place
  callout markers
- Export the calibration data as JSON for use in `src/callouts.py`

### 2D Replay Viewer (`replay.html`)

Tick-by-tick playback of an entire match on a 2D canvas.

- **All 10 players** rendered on the radar with team colors and health bars
- **Grenades**: smoke (18 s), molotov/incendiary (7 s), HE (2 s), flashbang
  (2 s) — each with flight path, activation radius, thrower label, and
  countdown timer
- **Kill markers**: skull icons at death positions with animated kill feed
  overlay
- **Playback controls**: play / pause, speed (0.5×–8×), round selector, tick
  scrubber
- Accessed from the Match Breakdown page

---

## Sync Folder

Instead of manually uploading each demo, you can point the app at your CS2
replays directory and selectively import new demos.

1. Open either the **Performance Breakdown** or **Match Breakdown** page.
2. Click the **Sync Folder** button in the sidebar.
3. **Configure the folder path** — the default path is the standard CS2
   replays location. Change it if your Steam library is elsewhere.
4. **Select an account** — the scan reads `.dem.info` sidecars to filter demos
   to matches the selected player actually participated in.
5. **Scan** — returns only demos not yet imported for that player.
6. **Select & process** — check the demos you want to import, click Process,
   and they are uploaded and analyzed in sequence with a progress bar.

---

## AI Features (Optional)

AI features require an API key from one of the supported providers. The app
works fully without AI — these features are additive.

### Supported Providers

| Provider | Example Models |
|----------|---------------|
| **OpenAI** | GPT-4.1, GPT-5.4-mini, O3, O4-mini |
| **Anthropic** | Claude Opus, Sonnet, Haiku |
| **Google Gemini** | Gemini 2.5, 3.1 |
| **Mistral** | Large, Medium, Small, Codestral |

### Setup

Configure via the API:

```bash
# Set your provider and API key
curl -X PUT http://localhost:8000/api/ai/config \
  -H "Content-Type: application/json" \
  -d '{
    "active_provider": "openai",
    "active_model": "gpt-4.1",
    "providers": {
      "openai": { "api_key": "sk-..." }
    }
  }'
```

Or edit `data/ai_config.json` directly (created on first use).

### What AI Powers

- **Match chat**: ask questions about any match — the system automatically
  builds a detailed context with round-by-round narratives including economy,
  kill/death positions with callouts, utility usage, clutch info, and trade
  details
- **AI assessment**: one request reads a map's positional data and its measured
  aim, utility and round behaviour together, naming both the roles played on
  each side and the habits behind the numbers
  (`POST /api/performance/ai-assessment?maps=de_mirage`). Run without a map, it
  becomes the career assessment: the same habits across every match, plus which
  maps the numbers say you actually perform on
- **Dashboard insights**: the performance page shows AI-generated tactical
  feedback

---

## What the Numbers Mean

Every figure the app shows has an entry in **[METRICS.md](METRICS.md)** — what it
measures, how it is worked out, and, where it grades you, where that grade came
from. The same text is in the app: the ⓘ beside a figure explains it in place.

Read the grading note there before treating a tier as a ranking. Most of the
cut-offs in this app are hand-set targets rather than percentiles of a real
player population, and the reference says which are which.

---

## Behavioral Assessment — How It Works

The 5-axis behavioral chart scores your playstyle on each side (CT and T)
across five dimensions. Every axis is scored **0–100** and computed from your
round-level data. The chart is shown per match in the Match Breakdown page and
aggregated in the Performance Breakdown.

### Axes

| Axis | What it measures | Key inputs |
|------|-----------------|------------|
| **Aggression** | How often you take or force the first duel of a round | Opening-duel involvement rate (% of rounds) and opening-kill win rate |
| **Trading** | How well you support teammates through trades and flashes | Trade-death %, flash assists per round, enemies flashed per round |
| **Isolation** | Tendency to play independently and pick off enemies without early confrontation | Survival % when *not* involved in the opening duel, non-involvement rate, kills per round |
| **Survival** | Ability to stay alive and contribute utility damage | Round survival %, utility damage per round (HE + Molotov), low-death rate |
| **Sniper** | Reliance on the AWP and long-range engagements | AWP kill ratio (% of all kills), long-range kill ratio (distance ≥ 30 units) |

### Scoring formulas (simplified)

- **Aggression** = `involvement_rate × 0.5 + opening_kill_% × 0.5` (capped at 100)
- **Trading** = `trade_death_% × 0.4 + flash_assists_pr × 50 (max 30) + enemies_flashed_pr × 25 (max 30)`
- **Isolation** = `survival_% × 0.4 + non_involvement_% × 0.3 + kills_pr × 40 (max 30)`
- **Survival** = `survival_% × 0.5 + util_dmg_pr × 3 (max 25) + (100 − death_rate) × 0.25`
- **Sniper** = `awp_kill_ratio × 0.7 + long_range_kill_ratio × 0.3`

### Dominant axis & success rate

Each round is also tagged with its **dominant behavior** — the axis that
scored highest in that specific round. The card then shows the **win rate**
for rounds where each axis was dominant, so you can see which playstyle
translates into actual round wins.

### Interpreting the chart

- A balanced pentagon means you're a versatile player with no extreme
  tendencies.
- A spike toward **Aggression** with low **Survival** often indicates an
  entry-fragger who creates space but dies frequently.
- High **Trading** + high **Survival** suggests a supportive anchor who stays
  alive while enabling teammates.
- A large **Sniper** axis with low **Trading** may indicate a passive AWPer
  who relies on picks instead of team play.
- Compare your CT chart to your T chart — most players have different profiles
  per side.

---

## API Reference

All endpoints are prefixed with `/api`. Interactive Swagger docs are at
**http://localhost:8000/docs**.

CORS is restricted to loopback — `localhost`, `127.0.0.1` and `[::1]`, on any
port. It is not open to all origins, and should not be: with credentials
allowed, that setting makes the browser hand your match history, your account
list and the Steam IDs of everyone you have played with to any website you
happen to have open while the app is running.

### Config & Accounts

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Return the active Steam ID |
| `PUT` | `/api/config` | Update the active Steam ID |
| `GET` | `/api/accounts` | List all accounts |
| `POST` | `/api/accounts` | Create an account (`name`, `steam_id`, optional `display_name`, `rank`) |
| `PUT` | `/api/accounts/{steam_id}` | Update account name, display_name, or rank |
| `PUT` | `/api/accounts/{steam_id}/activate` | Set an account as active |
| `DELETE` | `/api/accounts/{steam_id}` | Delete an account |

### Friends

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/friends` | List all friends |
| `POST` | `/api/friends` | Add a friend (`steam_id`, optional `name`) |
| `DELETE` | `/api/friends/{steam_id}` | Remove a friend |

### Matches

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/matches/upload` | Upload a `.dem` file (multipart: `file`, optional `info_file`, `steam_id`, `match_date`, `context_notes`, `tags`) |
| `POST` | `/api/matches/upload-bulk` | Upload multiple `.dem` files at once (multipart: `files`, optional `info_files`, `steam_id`) |
| `POST` | `/api/matches/detect-player` | Parse a `.dem.info` sidecar and match against known accounts |
| `GET` | `/api/matches` | List all matches; optional `?player_steam_id=` filter |
| `GET` | `/api/matches/{id}` | Full match detail: stats, teams, rounds, aim, utility, roles, behavioral axes |
| `PUT` | `/api/matches/{id}/notes` | Update context notes |
| `POST` | `/api/matches/{id}/tags` | Add a tag |
| `DELETE` | `/api/matches/{id}` | Delete a match and all related data |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/trends` | Trend data with averages; optional `?maps=dust2,mirage` filter |
| `GET` | `/api/performance` | Aggregated performance: HS%, side stats, opening duels, clutches, roles, multi-kills; optional `?maps=` |
| `POST` | `/api/performance/ai-assessment` | AI roles + playing patterns for a map (`?maps=de_mirage`); without `maps`, the career assessment over every match (patterns + map pool, no roles). Optional `&provider=&model=` |
| `GET` | `/api/performance/ai-assessment` | Get the persisted assessment (`?maps=de_mirage`, or no `maps` for the career one) |

### Minimap

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/matches/{id}/minimap` | Position data for minimap rendering; optional `?round_number=` |
| `POST` | `/api/minimap/zones` | Resolve callout names to pixel coordinates |
| `GET` | `/api/minimap/{map}/schematic` | All zone rectangles in pixel-space |

### AI Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/ai/providers` | List available AI providers and models |
| `GET` | `/api/ai/config` | Get current AI config (keys masked) |
| `PUT` | `/api/ai/config` | Update AI provider, model, API key, system instructions, prompts |
| `GET` | `/api/matches/{id}/chat` | Get chat history for a match |
| `POST` | `/api/matches/{id}/chat` | Send a message and receive an AI response |
| `DELETE` | `/api/matches/{id}/chat` | Clear chat history for a match |

### Sync Folder

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/sync/config` | Get current sync folder path |
| `PUT` | `/api/sync/config` | Update sync folder path |
| `GET` | `/api/sync/scan` | Scan for new `.dem` files; optional `?steam_id=` to filter by player |
| `POST` | `/api/sync/process` | Process selected demo files (JSON body: `{"files": [...]}`) |

### Storage

Demos are only needed until they are analysed — every feature, including the 2D
replay viewer, reads from the database afterwards. The one reason to keep a
demo is re-analysing it after changing how metrics are calculated, which is why
the default keeps a rolling window rather than deleting on import.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/storage/status` | Per-file classification: imported, kept, deletable, with sizes. Pass `keep_recent`, `per_account` or `fetched_only` to preview settings without saving them |
| `GET` | `/api/storage/config` | Retention settings |
| `PUT` | `/api/storage/config` | Update `keep_recent`, `per_account`, `fetched_only`, `auto_cleanup` |
| `POST` | `/api/storage/cleanup` | Delete demos outside the window (`{"dry_run": true}` to preview) |

A demo is only ever deleted when it is imported **and** its replay frames are
stored **and** it falls outside the retention window. With `fetched_only` (the
default) it must also be one this app downloaded, so demos you saved through CS2
are never touched.

`per_account` (default on) counts the window separately for each account, so a
heavily-played account cannot push another account's demos out of it. Turn it
off for a single global window, which bounds total disk use more tightly.

### Fetch from Steam *(optional)*

Setup (API key, per-account codes, QR sign-in) is in **Settings → Steam**. The
sidebar carries the two actions: **Manual Fetch from Steam** and **Auto-Sync**.

One Web API key covers every account — it is an API credential, not per-account
authentication (that is the per-account auth code). But a key is *issued* by one
specific Steam account, and nothing in the key says which, so **Settings → Steam**
records the issuing account alongside it and shows the key's last four
characters. That is what tells you whose profile to regenerate it from when it
stops working. `STEAM_API_KEY` in the environment overrides the stored key, and
the UI says so rather than showing a stale owner.

**Manual Fetch → Download Demos** lists every account the ledger knows about with
its outstanding count and a per-account tick box, so one run can cover a subset.
Excluding an account holds its matches back without forgetting them; the pending
total and the button label update as you tick. Whether an account is *tracked* at
all stays in **Settings → Steam**, since that decides if its history is recorded.

Requires the Node companion in `fetcher/`. Returns availability info rather than
failing when it is absent, so the app works normally without it.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/steam/status` | Setup state: Node present, accounts signed in / configured, matches outstanding |
| `PUT` | `/api/steam/api-key` | Store the Steam Web API key (one key covers all accounts) |
| `PUT` | `/api/steam/accounts/{name}` | Store an account's match-sharing auth code and starting share code |
| `POST` | `/api/steam/auth/{name}` | Start a QR sign-in; the QR image is returned via `/api/steam/job` events |
| `POST` | `/api/steam/check` | Look for matches played since the last check (no Steam session needed) |
| `POST` | `/api/steam/download` | Download demos for everything outstanding |
| `GET` | `/api/steam/job` | Poll the running (or last) job for live output |
| `POST` | `/api/steam/job/cancel` | Stop the running job (used when a QR sign-in is dismissed) |
| `GET` | `/api/steam/auto-sync` | Auto-Sync settings and live state |
| `PUT` | `/api/steam/auto-sync` | Turn Auto-Sync on/off, set the interval and the CS2 pause |
| `GET` | `/api/steam/presence` | Whether a tracked account is in CS2 right now |

#### Auto-Sync

The buttons above run one burst: check, then pull the whole backlog. Auto-Sync
is the same work at a trickle — **one match per cycle**, downloaded and then
analysed, with a gap in between (default 5 minutes; `0` runs them back to back).
Nothing is batched, so switching it off never leaves half a backlog in an
unknown state.

It shares the single job slot with the buttons, so the two never collide: a job
you start by hand takes priority and Auto-Sync waits for it.

**CS2 detection.** The app runs in a container and cannot see host processes, so
it asks Steam instead — `GetPlayerSummaries` reports `gameid` for accounts whose
*Game details* privacy is public. While a tracked account is in CS2, Auto-Sync
holds off: a 280 MB download mid-match costs you ping, and the fetcher's own
Game Coordinator login can drop you out of the game. If privacy hides the field
the UI says so and carries on rather than guessing. The behaviour is opt-out
(**Pause while CS2 is running**).

Auto-Sync never disables itself over an error — a Steam outage or one
unparseable demo should not silently end a background job you expect to still be
running. Failures back off exponentially (1 min → 30 min ceiling), and a demo
that fails to parse three times is set aside so the rest of the queue can carry
on. If it was left on, it resumes on startup.

### Replay

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/matches/{id}/replay` | Full tick-by-tick replay data (players, grenades, kills) |

### Upload example (curl)

```bash
curl -X POST http://localhost:8000/api/matches/upload \
  -F "file=@path/to/demo.dem" \
  -F "context_notes=ranked game" \
  -F "tags=solo queue,good game"
```

---

## Update Checks (Optional, off by default)

**Settings → Updates** shows the version you are running and, if you ask,
whether a newer one exists. Nothing about this is automatic on a fresh install.

There are two ways to ask. **Check now** looks once — pressing it is the
permission, so it works whatever the setting says. The tickbox above it turns on
a check once a day, and the answer is cached for 24 hours, so leaving it on
costs one request per day and works offline in between.

What the request is: a plain `GET` for `https://www.pr1maly.com/latest.json`, a
small static file listing the current version and a link to its release notes.
It carries nothing about your install — not your version, not an identifier, not
a count of anything — so what reaches the other end is an IP address and a
timestamp, which is true of any request to anything. The comparison happens on
your machine.

And that IP is not written down. The file is served from this project's own
server rather than a hosting service, and that server logs nothing for this one
path:

```nginx
location = /latest.json { access_log off; }
```

Which means nobody counts installs, including me. Handing the file to GitHub or
a CDN to serve would have moved the log rather than removed it; this way the
answer to "what happens to my address" is a line of configuration you can read
here rather than an assurance you have to take.

Nothing is ever installed for you. This application owns your match database and
migrates its schema when it starts; upgrading that unattended is a worse trade
than running a version behind. If an update exists you are shown the command for
how you installed it, and you run it when you choose.

`docker compose exec api cat /app/data/update_config.json` shows the setting,
and factory reset removes it — consent belongs to whoever gave it, not to the
next person using the machine.

---

## Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

The test suite covers the processor, database, and API layers.

---

## Wiping Data

Delete the SQLite database to start fresh:

```bash
# Linux / macOS
rm data/pr1mealazyer.db

# Windows
del data\pr1mealazyer.db
```

The database is recreated automatically on the next API start. Match demos are
not stored — only the extracted statistics.

---

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for what each part is responsible for.

```
pr1maly/
├── api.py                # Builds the app and registers the routers
├── pyproject.toml        # Package metadata, ruff / pytest / mypy config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example          # DEMO_HOST_DIR and friends
├── data/
│   ├── steamID           # Active Steam ID (legacy, managed by accounts)
│   ├── onboarding.json   # Onboarding wizard state
│   ├── accounts.json     # Multi-account configuration
│   ├── friends.json      # Friends list
│   ├── ai_config.json    # AI provider keys & settings
│   ├── ai_roles.json     # Persisted AI role assessments
│   ├── sync_config.json  # Sync folder path configuration
│   ├── storage_config.json    # Demo retention settings
│   ├── auto_sync.json         # Auto-Sync settings (survives restarts)
│   ├── steam_tokens.json      # Steam refresh tokens (fetcher, owner-only)
│   └── steam_sharecodes.json  # Match ledger + Web API key (fetcher, owner-only)
├── frontend/
│   ├── performance.html  # Main dashboard
│   ├── breakdown.html    # Aggregated performance breakdown
│   ├── match-breakdown.html  # Single match detail
│   ├── replay.html       # 2D tick-by-tick replay viewer
│   ├── calibrate.html    # Callout calibration tool
│   ├── charts.js         # Chart rendering, shared with the landing site
│   ├── theme.css / theme.js  # Dark/light theme support
│   ├── js/               # Scripts shared between pages
│   │   ├── accounts.js       # Account and friend management
│   │   └── steam-panel.js    # Fetch-from-Steam panel
│   ├── vendor/           # Tailwind + fonts, bundled so the app runs offline
│   ├── img/radar/        # Map radar images (1024×1024)
│   ├── img/maps/         # Map icons (512×512), named as in the DB
├── src/
│   ├── parser.py         # Demo parsing (Layer 1)
│   ├── processor.py      # Composes the metrics into one stored match
│   ├── database.py       # SQLite storage (Layer 3)
│   ├── callouts.py       # Map coordinate → callout translation
│   ├── ai_service.py     # Multi-provider AI integration
│   ├── api/              # HTTP layer
│   │   ├── deps.py           # Shared application state
│   │   ├── schemas.py        # Request / response models
│   │   └── routers/          # One module per group of endpoints
│   ├── domain/           # Pure logic — no HTTP, no database, no filesystem
│   │   ├── metrics/          # The measurements, each registered and versioned
│   │   │   ├── registry.py       # MetricSpec catalogue, served by GET /api/metrics
│   │   │   └── role_zones/       # Positional role definitions per map (JSON)
│   │   ├── calibration/      # Win probability, HLTV coefficients
│   │   ├── callouts/zones/   # Map callout rectangles (JSON)
│   │   └── blobs.py          # Decoding the stored JSON columns
│   ├── services/         # Imports, Steam jobs, Auto-Sync, retention, AI context
│   ├── metrics/behavior.py   # Cross-match behavioural axes and archetypes
│   └── config/           # Paths, and the JSON config store
├── tools/
│   └── build_release.py  # Builds the public source tree and landing site
├── fetcher/              # Node companion that downloads demos from Steam
└── tests/                # Including the API, schema and analysis snapshots
```

## Credits & Licensing

pr1maly's own code is covered by `LICENSE`. Everything else it bundles is listed
in `THIRD-PARTY-NOTICES`, which is the authoritative file — the summary below is
just a pointer.

**Counter-Strike 2 assets.** `frontend/img/radar/` and `frontend/img/maps/`
contain map radar images and map icons that are the property of Valve
Corporation. They are not covered by the pr1maly license and are not licensed
for redistribution. The radars come from the game's own `resource/overviews/`
directory; the icons were retrieved via
[cs2-map-icons](https://github.com/MurkyYT/cs2-map-icons), which extracts them
from Valve's public depot — credited as the source, not as a licensor.

Counter-Strike and Valve are trademarks of Valve Corporation. pr1maly is not
affiliated with, endorsed by, or sponsored by Valve Corporation.

**Libraries.** demoparser2, pandas, FastAPI, uvicorn, Tailwind CSS, Chart.js,
Material Symbols, SQLite and Python — each with its license in
`THIRD-PARTY-NOTICES`.
