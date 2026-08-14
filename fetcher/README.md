# Demo Fetcher

Pulls your CS2 match demos straight from Valve so you don't have to download
them in-game first.

It writes `.dem` files (plus a `.dem.info` sidecar) into your CS2 replays
folder. From there the app's normal **Sync Folder** flow imports them — nothing
in `api.py` or the frontend changes.

It ships with the release. `tools/build_release.py` copies the source and the
lockfile, never `node_modules/` (npm rebuilds that) and never `data/` — your
Steam tokens and match ledger stay on the machine that made them.

It is optional: the app works without it, and every other import route still
exists. What it buys you is not having to save each demo in-game.

---

## Requirements

- Node.js 18+ (tested on 24)
- The accounts you want must already exist in `data/accounts.json` with a Steam ID

```bash
cd fetcher
npm install
```

---

## Two modes

| | **ledger** (share codes) | **recent** (own history) |
|---|---|---|
| Setup | API key + 2 codes per account | none |
| Matches visible | all of them, indefinitely | a small fixed window |
| Accounts covered per run | all, via one resolver | one login per account |
| Steam client conflict | only the resolver account | every account in turn |

`fetch.js` uses the ledger whenever it has pending matches, otherwise recent.
Force it with `--mode ledger` or `--mode recent`.

Recent mode needs no setup and is fine for one account. Ledger mode is what you
want with several, because **one account can resolve matches played by any
other** — so the account your Steam client is holding no longer blocks you.

---

## Step 1 — authenticate (both modes)

**From the app**: open **Fetch from Steam** in the sidebar and press *sign in*
next to an account. A QR code appears in the browser; scan it with the Steam
mobile app. Nothing secret is typed — a QR login is a device authorisation, not
a password prompt, which is why it is safe to expose in the UI.

**From a terminal**, if you prefer, or if you want the password path:

```bash
npm run auth -- pr1me --qr      # same QR flow, drawn in the terminal
npm run auth -- pr1me           # password + Steam Guard
npm run status                  # see who is authenticated
```

QR is easier either way: no username, no password, and it can't fail on a name
mismatch. The password flow is terminal-only on purpose — a Steam password has
no business in a web form.

If you use the credential path, note that the account label in
`data/accounts.json` is something *you* typed in the app and is often not your
Steam **login** name. `auth.js` prompts for it, or pass it directly:

```bash
npm run auth -- pr1me --login my_actual_steam_login
```

Steam returns `InvalidPassword` for a wrong *username* as well as a wrong
password, so that error usually means the login name.

---

## Step 2 — set up the ledger (optional, recommended)

**One Web API key, total.** It identifies the caller, not the account being
queried, so a single key covers all your accounts. Get it from
<https://steamcommunity.com/dev/apikey> using whichever account has spent $5
(limited accounts can't have one).

```bash
node sharecodes.js --api-key ABCDEF0123456789
```

Or set `STEAM_API_KEY` in the environment and skip storing it.

**Two codes per account.** Open Steam's CS2 game-data page in a browser while
signed in as that account:

<https://help.steampowered.com/en/wizard/HelpWithGameIssue/?appid=730&issueid=128>

It shows a match-sharing authentication code and your most recent share code.

```bash
node sharecodes.js --set pr1me \
  --auth-code ABCD-EFGHI-JKLM \
  --share-code CSGO-abcde-fghij-klmno-pqrst-uvwxy
```

That's browser login, not a client session, so it never conflicts with a
running Steam. The auth code doesn't expire unless you regenerate it. The share
code is only a starting point — the ledger walks forward from there on its own.

Repeat per account, then:

```bash
npm run walk        # ask Steam what has been played since the cursor
npm run codes -- --status
```

### Choosing which accounts do what

Tracking and downloading are independent. An account can stay in the ledger —
so you always know what it played — without its demos being fetched.

```bash
node sharecodes.js --toggle Penetreto --download off   # track, don't download
node sharecodes.js --toggle Farruhlkt --walk off       # ignore entirely
```

Both are checkboxes on each account row in **Fetch from Steam**. Neither
affects which account *resolves* share codes — any signed-in account can do
that, including one you have downloading switched off.

Walking needs no Steam session at all. You can do it mid-match.

---

## Step 3 — fetch

```bash
npm start                        # ledger if it has work, else recent
node fetch.js --limit 5          # only the newest 5 outstanding matches
node fetch.js --dry-run          # show what would happen, download nothing
node fetch.js --resolver pr1me   # force which account resolves share codes
node fetch.js --mode recent      # ignore the ledger
node fetch.js --repair-sidecars  # rewrite sidecars for demos already on disk
node fetch.js --debug            # dump the raw GC response for one match
```

`--limit` takes the **newest** N, not the oldest. A first run over a long
history would otherwise pull tens of gigabytes, most of it matches old enough
that Valve has already deleted the demo. In the UI it is the *Download at most*
box; leave it blank for everything.

Then open the app and click **Sync Folder** to import.

A typical week is `npm run walk` followed by `npm start`.

### About map names

The Game Coordinator usually returns no map for historical matches —
`watchablematchinfo` describes matches you could *watch live*, so it's empty
for finished ones. Rather than write a placeholder, the sidecar omits the map
entirely, which makes the app read it from the demo header via
`read_demo_map()`. That's authoritative anyway.

So demos may be named `pr1maly_<time>_<id>.dem` with no map in the filename.
The app still shows the correct map.

---

## One client session per account

Steam permits **one game-client session per account**. The fetcher is a game
client, so it collides only with the account your Steam desktop client is
signed into:

| Steam client signed into | Fetching as | Result |
|---|---|---|
| `pr1me` | `pr1me` | collides |
| `pr1me` | `brabrabrupt` | fine |
| nothing / Steam closed | any | fine |

Not a per-device limit — Steam and the fetcher coexist happily on one machine
as long as they're different accounts. Web and mobile-app sessions use a
different session type and never conflict.

**In ledger mode** this rarely matters: the fetcher tries each authenticated
account in turn until one gets a session, and whichever wins can resolve
everyone's matches.

**With only one account set up** there is no alternative, so a collision is
fatal for that run and the error says so:

```
"pr1me" is logged in elsewhere, and it is the only account set up,
so there is no other account available to fetch with.

Quit the Steam desktop client completely: right-click the tray icon and
choose Exit. Closing the window is not enough.
```

Closing the window leaves Steam in the tray with the session still claimed.

---

## Two limits worth knowing

**Recent mode sees a fixed-size window**, not a date range — the Game
Coordinator returns only your most recent handful of matches. Play more than
that between runs and the older ones become invisible. Ledger mode has no such
limit.

**Demos expire.** Valve deletes them from `replay*.valve.net` after roughly a
month — measured on this setup, a 23-day-old match downloaded fine while a
34-day-old one was already gone. Expired matches report `gone` or `no demo
URL`, and are unrecoverable by any method. Anything already in your replays
folder is safe permanently.

The CDN answers `HTTP 502` for demos it no longer holds. Since 502 can also be
a transient fault, it's only treated as permanent once the match is older than
28 days; younger ones stay pending and are retried on the next run.

The ledger records that a match existed even when its demo is gone, so you can
always tell a quiet week from a broken fetcher.

---

## Running under Docker

The root `Dockerfile` installs Node and the fetcher's packages into the API
image, so **Fetch from Steam** works from the web UI with no extra setup:

```bash
docker compose up --build
```

Two things the compose file handles for you:

- the demo folder is mounted **`:rw`**, because the fetcher writes to it
  (Sync Folder alone only ever read from it)
- `DEMO_DIR=/demos` points the fetcher at that mount

Authentication still happens on the host, not in the container — `data/` is a
bind mount, so a token created with `npm run auth` on your machine is visible
inside the container immediately. Run `npm install` locally once if you want to
use the CLI there too.

Both ship as they are. The fetcher is part of the release, so the image needs
Node and the demo folder has to stay writable — it is where downloaded demos
land for Sync Folder to pick up.

---

## Where demos are written

First path that exists on disk wins:

1. `DEMO_DIR` environment variable
2. `folder` in `data/sync_config.json`
3. the default CS2 replays path

`sync_config.json` holds the *container* path (`/demos`) under Docker, which is
why it's skipped when it doesn't resolve locally.

```powershell
$env:DEMO_DIR = "D:\cs2-demos"
npm start
```

---

## Security

Two files under `data/` are sensitive, and `data/` is already gitignored:

- `steam_tokens.json` — **refresh tokens**, password-equivalent and long-lived
- `steam_sharecodes.json` — your Web API key and per-account auth codes

Both are written owner-only. Don't paste either anywhere.

To revoke tokens: change your Steam password, or use *Deauthorize all devices*.
Both invalidate every stored token at once, so you'd re-run `npm run auth` for
each account. Neither affects the ledger.

```bash
node auth.js --forget pr1me      # drop one stored token locally
```

---

## Troubleshooting

| Message | Meaning |
|---|---|
| `is logged in elsewhere` | Steam client holds that account. Use another account, or quit Steam fully. Token is fine. |
| `stored token is no longer valid` | Token expired or revoked. Re-run `npm run auth -- <account>`. |
| `InvalidPassword` during auth | Usually the wrong *login name*, not the password. Use `--qr` or `--login`. |
| `Steam rejected the request (403)` | Bad API key, or the auth code doesn't match that account's Steam ID. |
| `Steam rejected the known share code (412)` | Stored share code isn't valid for that account. Re-seed with `--set`. |
| `Rate limited by Steam (429)` | Wait a few minutes. |
| `interrupted, retry 1/2...` | The CDN dropped the connection mid-download. Retried automatically with a growing pause; usually succeeds on the next attempt. |
| `FAILED (terminated)` | All retries were interrupted. The match stays pending and is retried on the next run. |
| `gone` / `no demo URL` | Demo deleted by Valve. Unrecoverable. |
| `No usable demo output folder found` | Set `DEMO_DIR` explicitly. |
