# pr1maly — CS2 Local Analytics & Trend Tracker

A standalone, locally-hosted application for Counter-Strike 2. Upload `.dem`
files, extract advanced player metrics, and track performance trends over
time — all stored locally with no external services required.

## Features

- **HLTV 2.0 Rating, ADR, KAST%, K/D, Impact, KPR/DPR** — HP-accurate ADR
  (no overkill, matches Refrag/Leetify)
- **Multi-kill rounds** (2K–5K), **clutch detection** (1vX), **trade detection**
- **Full 10-player scoreboard** with team split and rank extraction
- **Round-by-round enriched data**: economy, buy type, kill/death callout
  positions, opening duels, bomb events, utility usage
- **Aim & movement analysis**: aim score (0–100), counterstrafe detection,
  stance breakdown, per-weapon movement penalties, reaction time, crosshair
  placement quality, time-to-kill efficiency
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
- **Three upload methods**: single, bulk, and folder sync
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

| Layer | Purpose | Tech |
|-------|---------|------|
| **1 – Parser** | Reads raw `.dem` files into DataFrames | `demoparser2` |
| **2 – Processor** | Filters events by Steam ID, calculates stats | `pandas` |
| **3 – Storage** | Persists matches, round timelines, tags, players | `sqlite3` |
| **4 – API** | REST endpoints + serves the frontend | `FastAPI`, `uvicorn` |
| **5 – Frontend** | Interactive HTML/JS pages | Vanilla JS, Chart.js |
| **6 – AI** | Match coaching & role assessment | OpenAI / Anthropic / Gemini / Mistral |

---

## Quick Start

### Prerequisites

- **Python 3.11+** (tested with 3.11 and 3.14)
- **pip** (comes with Python)
- **Docker** and **Docker Compose** *(only needed for Option B)*

### Option A — Run locally (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/christianhefti/pr1mealazyer.git
cd pr1mealazyer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser. The app redirects you to the
Performance dashboard automatically.

### Option B — Run with Docker

```bash
# 1. Build and start
docker compose up --build

# 2. Open http://localhost:8000
# 3. Stop with:
docker compose down
```

The SQLite database is persisted in `./data/` on your host via a Docker volume.

> **Sync Folder**: the Docker Compose file mounts the default CS2 replays
> directory (`C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays`)
> as `/demos` inside the container. Edit the path in `docker-compose.yml` if
> your Steam library lives elsewhere.

---

## First-Time Setup

1. **Open the app** at **http://localhost:8000**. You'll land on the
   Performance dashboard.

2. **Configure your Steam ID** — you have two options:
   - **Via file**: create `data/steamID` containing your 64-bit Steam ID on a
     single line (find yours at [steamid.io](https://steamid.io)):
     ```
     76561197971699749
     ```
   - **Via API**: call `PUT /api/config` with `{"steam_id": "76561197971699749"}`.
   - **Via accounts API**: create an account with
     `POST /api/accounts` (see API section below). The first account is
     activated automatically.

3. **Download a demo** from CS2: go to *CS2 → Profile → Matches* and download
   the `.dem` file for the match you want to analyze.

4. **Upload the demo**:
   - **curl**:
     ```bash
     curl -X POST http://localhost:8000/api/matches/upload \
       -F "file=@path/to/demo.dem" \
       -F "context_notes=ranked game" \
       -F "tags=solo queue,good game"
     ```
   - **Swagger UI**: open **http://localhost:8000/docs**, find the
     `/api/matches/upload` endpoint, and use the "Try it out" form to select
     your `.dem` file.
   - **Bulk upload**: send multiple `.dem` files at once to
     `POST /api/matches/upload-bulk`. Pair each demo with its `.dem.info`
     sidecar for automatic date detection and account matching.
   - **Sync folder**: click the Sync Folder button on the Breakdown or Match
     Breakdown page to scan your CS2 replays directory and selectively import
     new demos (see the Sync Folder section below).

5. **Browse your stats** — every page updates automatically once matches are
   uploaded.

---

## Frontend Pages

The frontend is served at `/frontend/` and consists of five pages. Navigate
between them using the sidebar.

### Performance Dashboard (`performance.html`)

The main landing page showing your overall career stats.

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
- **AI role assessment**: request AI to analyze your CT and T roles on a
  specific map based on all your positioning data (`POST /api/performance/ai-roles?maps=de_mirage`)
- **Dashboard insights**: the performance page shows AI-generated tactical
  feedback

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

All endpoints are prefixed with `/api`. CORS is enabled for all origins.
Interactive Swagger docs are at **http://localhost:8000/docs**.

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
| `POST` | `/api/performance/ai-roles` | AI-powered role assessment for a map (`?maps=de_mirage`) |
| `GET` | `/api/performance/ai-roles` | Get persisted AI role assessments (`?maps=de_mirage`) |

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

```
pr1mealazyer/
├── api.py                # FastAPI REST backend (Layer 4)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── steamID           # Active Steam ID (auto-managed)
│   ├── accounts.json     # Multi-account configuration
│   ├── friends.json      # Friends list
│   ├── ai_config.json    # AI provider keys & settings
│   ├── ai_roles.json     # Persisted AI role assessments
│   └── sync_config.json  # Sync folder path configuration
├── frontend/
│   ├── performance.html  # Main dashboard
│   ├── breakdown.html    # Aggregated performance breakdown
│   ├── match-breakdown.html  # Single match detail
│   ├── replay.html       # 2D tick-by-tick replay viewer
│   ├── calibrate.html    # Callout calibration tool
│   ├── theme.css / theme.js  # Dark/light theme support
│   ├── img/radar/        # Map radar images (1024×1024)
│   └── txt/maps/         # Map overview config files
├── src/
│   ├── parser.py         # Demo parsing (Layer 1)
│   ├── processor.py      # Metrics calculation (Layer 2)
│   ├── database.py       # SQLite storage (Layer 3)
│   ├── callouts.py       # Map coordinate → callout translation
│   └── ai_service.py     # Multi-provider AI integration
└── tests/
    ├── test_api.py
    ├── test_processor.py
    └── test_database.py
```
