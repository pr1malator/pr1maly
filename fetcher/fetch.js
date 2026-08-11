'use strict';

// Downloads new CS2 match demos.
//
//   node fetch.js                     use the best available mode
//   node fetch.js --mode ledger       resolve share codes from the match ledger
//   node fetch.js --mode recent       ask each account for its own recent games
//   node fetch.js --resolver pr1me    force which account resolves share codes
//   node fetch.js --account pr1me     recent mode: only this account
//   node fetch.js --limit 5           only the newest 5 outstanding matches
//   node fetch.js --dry-run           list what would happen, download nothing
//
// Two ways to find matches:
//
//   ledger  Share codes collected by sharecodes.js are resolved through the
//           Game Coordinator by ONE account, which can resolve matches played
//           by any other account. No fixed-size window, and only the resolver
//           account can collide with a running Steam client.
//
//   recent  Each account asks for its own recent match history. Needs no
//           setup, but returns only a small fixed window and requires logging
//           in as every account in turn.
//
// Demos land in your CS2 replays folder with a .dem.info sidecar, so the app's
// existing Sync Folder scan imports them unchanged.

const fs = require('fs');
const path = require('path');

const SteamUser = require('steam-user');
const GlobalOffensive = require('globaloffensive');

const { loadAccounts, resolveOutputDir } = require('./lib/paths');
const tokens = require('./lib/tokens');
const ledger = require('./lib/ledger');
const { buildDemInfo } = require('./lib/deminfo');
const { downloadWithRetry } = require('./lib/download');

// Cancelling from the UI arrives as SIGTERM. Abort the in-flight download so
// its partial files are cleaned up instead of being left in the demo folder.
// (Windows kills the process outright and never delivers this; the stale-file
// sweep below covers that case.)
const cancellation = new AbortController();
process.on('SIGTERM', () => {
  console.log('\nCancelled.');
  cancellation.abort();
  setTimeout(() => process.exit(130), 2_000).unref();
});

const CS2_APPID = 730;
const CONNECT_TIMEOUT_MS = 90_000;
const MATCHLIST_TIMEOUT_MS = 45_000;
const RESOLVE_TIMEOUT_MS = 20_000;
const PAUSE_BETWEEN_ACCOUNTS_MS = 3_000;
const PAUSE_BETWEEN_RESOLVES_MS = 1_500;

// Valve's replay CDN returns 502 for demos it no longer holds. Past this age a
// 502 is treated as permanent rather than retried forever. Observed retention
// is roughly 30 days.
const EXPIRY_GRACE_DAYS = 28;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─── error classification ────────────────────────────────────────────────────

/** Steam results that mean "this refresh token is dead, log in again". */
function isDeadTokenError(err) {
  const E = SteamUser.EResult;
  const dead = [E.InvalidPassword, E.AccessDenied, E.Expired, E.Revoked].filter(
    (v) => v !== undefined
  );
  return dead.includes(err.eresult);
}

/**
 * Steam allows one game-client session per account. If the desktop client
 * already holds it, our logon loses. The stored token is fine — this is a
 * scheduling problem, not an auth one.
 */
function isSessionConflict(err) {
  const E = SteamUser.EResult;
  const conflict = [E.LoggedInElsewhere, E.LogonSessionReplaced].filter((v) => v !== undefined);
  return conflict.includes(err.eresult);
}

/**
 * Wording depends on whether another account could have done the job instead.
 * With a single account there is no alternative, so closing Steam is the only
 * way forward and the message says so plainly.
 */
function sessionConflictMessage(account, { alternativesAvailable }) {
  const quitSteam =
    '      Quit the Steam desktop client completely: right-click the tray icon and\n' +
    '      choose Exit. Closing the window is not enough — it keeps running in the\n' +
    '      tray and the session stays claimed.\n\n' +
    '      Your stored token is still valid. Do NOT re-authenticate.';

  if (!alternativesAvailable) {
    return (
      `"${account.name}" is logged in elsewhere, and it is the only account set up,\n` +
      '      so there is no other account available to fetch with.\n\n' +
      quitSteam
    );
  }

  return (
    `"${account.name}" is logged in elsewhere — the Steam client is holding it.\n\n` +
    '      Either fetch with a different account, or:\n\n' +
    quitSteam
  );
}

// ─── steam / game coordinator ────────────────────────────────────────────────

/** Log in and connect to the CS2 Game Coordinator. */
function connect(client, csgo, account, refreshToken, { alternativesAvailable }) {
  return new Promise((resolve, reject) => {
    let settled = false;

    const done = (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (err) reject(err);
      else resolve();
    };

    const timer = setTimeout(
      () =>
        done(
          new Error(
            `Timed out after ${CONNECT_TIMEOUT_MS / 1000}s waiting for the CS2 Game ` +
              'Coordinator. It is occasionally slow or down — try again in a few minutes.'
          )
        ),
      CONNECT_TIMEOUT_MS
    );

    client.once('loggedOn', () => {
      // Appearing "in game" is what gets us a GC session.
      client.gamesPlayed([CS2_APPID]);
    });

    csgo.once('connectedToGC', () => done());

    client.on('error', (err) => {
      if (isSessionConflict(err)) {
        const conflict = new Error(sessionConflictMessage(account, { alternativesAvailable }));
        conflict.sessionConflict = true;
        done(conflict);
      } else if (isDeadTokenError(err)) {
        done(
          new Error(
            `stored token is no longer valid (${err.message}). Re-authenticate with:\n` +
              `      npm run auth -- ${account.name}`
          )
        );
      } else {
        done(err);
      }
    });

    client.logOn({ refreshToken });
  });
}

function openSession(account, entry, options) {
  const client = new SteamUser({ autoRelogin: false });
  const csgo = new GlobalOffensive(client);

  const close = () => {
    client.removeAllListeners('error');
    client.on('error', () => {}); // swallow disconnect noise during teardown
    try {
      client.logOff();
    } catch {
      /* already disconnected */
    }
  };

  return connect(client, csgo, account, entry.refreshToken, options).then(
    () => ({ client, csgo, close }),
    (err) => {
      close();
      throw err;
    }
  );
}

/** Ask the GC for this account's own recent match history. */
function requestRecentGames(csgo) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      csgo.removeListener('matchList', onMatchList);
      reject(new Error('Timed out waiting for match history from the GC.'));
    }, MATCHLIST_TIMEOUT_MS);

    function onMatchList(matches) {
      clearTimeout(timer);
      resolve(matches || []);
    }

    csgo.once('matchList', onMatchList);
    csgo.requestRecentGames();
  });
}

/** Resolve one share code. Works for matches played by any account. */
function requestGameByShareCode(csgo, shareCode) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      csgo.removeListener('matchList', onMatchList);
      reject(new Error('the Game Coordinator did not answer'));
    }, RESOLVE_TIMEOUT_MS);

    function onMatchList(matches) {
      clearTimeout(timer);
      resolve((matches && matches[0]) || null);
    }

    csgo.once('matchList', onMatchList);
    csgo.requestGame(shareCode);
  });
}

// ─── match handling ──────────────────────────────────────────────────────────

/**
 * Pull the fields we care about out of a GC match record.
 * The demo URL lives on the final roundstats entry, in a field named "map".
 */
function describeMatch(match) {
  const rounds = match.roundstatsall || [];
  const last = rounds[rounds.length - 1] || {};
  const watchable = match.watchablematchinfo || {};

  // watchablematchinfo describes a *live* match, so historical results often
  // carry no map at all. mapgroup ("mg_de_mirage") is the usual second chance.
  let mapName = watchable.game_map || null;
  if (!mapName && typeof watchable.game_mapgroup === 'string') {
    mapName = watchable.game_mapgroup.replace(/^mg_/, '');
  }
  if (mapName) mapName = String(mapName).trim() || null;

  return {
    matchId: String(match.matchid),
    matchTime: Number(match.matchtime) || 0,
    mapName, // null when the GC didn't tell us — the app reads it from the demo
    demoUrl: last.map || null,
    playerIds: (last.reservation && last.reservation.account_ids) || [],
  };
}

/**
 * Name demos so the app can still recover the date from the filename if the
 * sidecar is ever missing — sync_scan looks for _<10-digit unix time>_.
 * The map is included only when known, never as a placeholder.
 */
function demoFilename(info) {
  const parts = ['pr1maly'];
  if (info.mapName) parts.push(info.mapName);
  parts.push(info.matchTime, info.matchId);
  return `${parts.join('_')}.dem`;
}

function matchLabel(info) {
  const date = info.matchTime
    ? new Date(info.matchTime * 1000).toISOString().slice(0, 10)
    : 'unknown date';
  // The GC rarely reports a map for historical matches; the app reads it from
  // the demo instead, so just omit it here rather than printing a placeholder.
  return info.mapName ? `${info.mapName} ${date}` : date;
}

function ageInDays(matchTime) {
  if (!matchTime) return Infinity;
  return (Date.now() / 1000 - matchTime) / 86_400;
}

/** Download a demo, retrying transient interruptions. See lib/download.js. */
function fetchDemo(url, destPath) {
  return downloadWithRetry(url, destPath, {
    signal: cancellation.signal,
    onRetry: (attempt, maxRetries, err) =>
      process.stdout.write(`${err.message}, retry ${attempt}/${maxRetries}... `),
  });
}

/**
 * Remove temp files from a run that was killed part-way through.
 * .bz2 files are left alone — a failed decode keeps one on purpose.
 */
function sweepPartialFiles(outputDir) {
  let removed = 0;
  for (const name of fs.readdirSync(outputDir)) {
    if (!name.endsWith('.part')) continue;
    try {
      fs.rmSync(path.join(outputDir, name), { force: true });
      removed += 1;
    } catch {
      /* in use, or gone already */
    }
  }
  if (removed) console.log(`Cleaned up  : ${removed} partial file(s) from an interrupted run`);
}

/**
 * Download a match and write its sidecar.
 * @returns {Promise<{status: string, filename: string, bytes?: number, reason?: string}>}
 */
async function saveMatch(info, ownerSteamId, outputDir) {
  const filename = demoFilename(info);
  const destPath = path.join(outputDir, filename);

  if (fs.existsSync(destPath)) return { status: 'downloaded', filename, bytes: 0, existing: true };

  try {
    const startedAt = Date.now();
    const bytes = await fetchDemo(info.demoUrl, destPath);
    const seconds = (Date.now() - startedAt) / 1000;

    writeSidecar(destPath, info, ownerSteamId);
    return { status: 'downloaded', filename, bytes, seconds };
  } catch (err) {
    // 403/404 are unambiguous. Valve's replay CDN answers 502 for demos it no
    // longer holds, but 502 is also a plausible transient fault — so only call
    // it expired once the match is old enough that recovery is hopeless.
    if (err.status === 403 || err.status === 404) {
      return { status: 'expired', filename, reason: 'deleted from Valve servers' };
    }
    if (err.status === 502 && ageInDays(info.matchTime) > EXPIRY_GRACE_DAYS) {
      return {
        status: 'expired',
        filename,
        reason: `HTTP 502, ${Math.round(ageInDays(info.matchTime))} days old — past retention`,
      };
    }
    // tagError already produced a message naming the phase that failed.
    return { status: 'failed', filename, reason: err.message };
  }
}

/**
 * Write the .dem.info sidecar: date, players, and the map when we know it.
 *
 * The map is omitted rather than guessed. sync_scan treats any non-empty
 * map_name as authoritative, so a placeholder would stick — leaving it out
 * makes the app fall back to reading the real map from the demo header.
 */
function writeSidecar(destPath, info, ownerSteamId) {
  const sidecar = buildDemInfo({
    matchTime: info.matchTime,
    mapName: info.mapName || null,
    steamIds: [...info.playerIds, ownerSteamId].filter(Boolean),
  });
  fs.writeFileSync(`${destPath}.info`, sidecar);
}

// ─── mode: ledger (share codes) ──────────────────────────────────────────────

async function runLedgerMode(accounts, stored, outputDir, { dryRun, resolverName, debug, limit }) {
  const state = ledger.load();
  let pending = ledger.pendingMatches(state);
  const outstanding = pending.length;
  const totals = { downloaded: 0, skipped: 0, expired: 0, failed: 0 };

  // pendingMatches is in play order, so the tail is the most recent. Those are
  // the ones worth having: anything old enough is likely expired regardless.
  if (limit > 0 && pending.length > limit) {
    pending = pending.slice(-limit);
  }

  console.log(
    pending.length < outstanding
      ? `Mode        : ledger (newest ${pending.length} of ${outstanding} pending)`
      : `Mode        : ledger (${pending.length} match(es) pending)`
  );
  if (!pending.length) {
    console.log('\nNothing pending. Run "node sharecodes.js --walk" to look for new matches.');
    return totals;
  }

  const byName = new Map(accounts.map((a) => [a.name, a]));
  let candidates = accounts.filter((a) => stored[a.name]);

  if (resolverName) {
    candidates = candidates.filter((a) => a.name.toLowerCase() === resolverName.toLowerCase());
    if (!candidates.length) throw new Error(`"${resolverName}" has no stored token.`);
  }

  if (!candidates.length) {
    throw new Error('No authenticated accounts to resolve share codes with.');
  }

  if (dryRun) {
    for (const item of pending) {
      console.log(`  + ${item.label}: ${item.shareCode} would be resolved and downloaded`);
    }
    totals.downloaded = pending.length;
    return totals;
  }

  // Any authenticated account can resolve any share code, so fall through the
  // list until one manages to hold a Game Coordinator session.
  let session = null;
  let resolver = null;
  const failures = [];

  for (const [index, account] of candidates.entries()) {
    try {
      process.stdout.write(`Resolver    : trying ${account.name}... `);
      session = await openSession(account, stored[account.name], {
        alternativesAvailable: index < candidates.length - 1,
      });
      resolver = account;
      console.log('connected');
      break;
    } catch (err) {
      console.log('unavailable');
      failures.push({ account, err });
    }
  }

  if (!session) {
    const conflict = failures.find((f) => f.err.sessionConflict);
    throw new Error(
      conflict
        ? conflict.err.message
        : failures.map((f) => `${f.account.name}: ${f.err.message}`).join('\n      ')
    );
  }

  try {
    for (const item of pending) {
      const owner = byName.get(item.label);
      const ownerSteamId = owner ? owner.steam_id : null;

      try {
        const match = await requestGameByShareCode(session.csgo, item.shareCode);
        if (!match) {
          console.log(`  - ${item.label} ${item.shareCode}: no match data returned`);
          const fresh = ledger.load();
          ledger.markMatch(fresh, item.label, item.shareCode, 'failed');
          ledger.save(fresh);
          totals.failed += 1;
          continue;
        }

        if (debug) {
          const dumpPath = path.join(__dirname, `gc-match-${item.shareCode}.json`);
          fs.writeFileSync(dumpPath, JSON.stringify(match, null, 2));
          console.log(`  [debug] raw GC response written to ${dumpPath}`);
        }

        const info = describeMatch(match);
        if (!info.demoUrl) {
          console.log(`  - ${matchLabel(info)}: no demo URL (expired)`);
          const fresh = ledger.load();
          ledger.markMatch(fresh, item.label, item.shareCode, 'expired');
          ledger.save(fresh);
          totals.expired += 1;
          continue;
        }

        process.stdout.write(`  + ${item.label} ${matchLabel(info)}: `);
        const result = await saveMatch(info, ownerSteamId, outputDir);

        const fresh = ledger.load();
        ledger.markMatch(fresh, item.label, item.shareCode, result.status, {
          filename: result.filename,
        });
        ledger.save(fresh);

        if (result.status === 'downloaded') {
          console.log(
            result.existing
              ? 'already on disk'
              : `${(result.bytes / 1024 / 1024).toFixed(1)} MB in ${result.seconds.toFixed(0)}s`
          );
          if (result.existing) totals.skipped += 1;
          else totals.downloaded += 1;
        } else if (result.status === 'expired') {
          console.log(`gone (${result.reason})`);
          totals.expired += 1;
        } else {
          console.log(`FAILED (${result.reason})`);
          totals.failed += 1;
        }
      } catch (err) {
        console.log(`  ! ${item.label} ${item.shareCode}: ${err.message}`);
        const fresh = ledger.load();
        ledger.markMatch(fresh, item.label, item.shareCode, 'failed');
        ledger.save(fresh);
        totals.failed += 1;
      }

      await sleep(PAUSE_BETWEEN_RESOLVES_MS);
    }
  } finally {
    session.close();
  }

  console.log(`\nResolved using "${resolver.name}".`);
  return totals;
}

// ─── mode: recent games ──────────────────────────────────────────────────────

async function runRecentMode(accounts, stored, outputDir, { dryRun, only, limit }) {
  let targets = accounts.filter((a) => stored[a.name]);

  if (only) {
    targets = targets.filter((a) => a.name.toLowerCase() === only.toLowerCase());
    if (!targets.length) throw new Error(`"${only}" has no stored token. Run: npm run auth -- ${only}`);
  }

  if (!targets.length) {
    throw new Error(
      'No authenticated accounts. Authenticate one first:\n' +
        '  npm run auth -- <account-name>\n\n' +
        'See which accounts are known with: npm run status'
    );
  }

  console.log(`Mode        : recent games (${targets.length} account(s))`);

  const totals = { downloaded: 0, skipped: 0, expired: 0, failed: 0 };
  const problems = [];

  for (const [index, account] of targets.entries()) {
    console.log(`\n[${index + 1}/${targets.length}] ${account.name} (${account.steam_id})`);

    let session = null;
    try {
      // In this mode every account fetches its own history, so another account
      // being available does not help if this one is blocked.
      session = await openSession(account, stored[account.name], {
        alternativesAvailable: false,
      });

      const all = await requestRecentGames(session.csgo);
      const matches = limit > 0 ? all.slice(-limit) : all;
      console.log(
        matches.length < all.length
          ? `  Game Coordinator returned ${all.length}; taking the newest ${matches.length}.`
          : `  Game Coordinator returned ${matches.length} recent match(es).`
      );

      for (const raw of matches) {
        const info = describeMatch(raw);

        if (!info.demoUrl) {
          console.log(`  - ${matchLabel(info)}: no demo URL (too old or still processing)`);
          totals.expired += 1;
          continue;
        }

        if (fs.existsSync(path.join(outputDir, demoFilename(info)))) {
          totals.skipped += 1;
          continue;
        }

        if (dryRun) {
          console.log(`  + ${matchLabel(info)}: would download -> ${demoFilename(info)}`);
          totals.downloaded += 1;
          continue;
        }

        process.stdout.write(`  + ${matchLabel(info)}: `);
        const result = await saveMatch(info, account.steam_id, outputDir);

        if (result.status === 'downloaded') {
          console.log(`${(result.bytes / 1024 / 1024).toFixed(1)} MB in ${result.seconds.toFixed(0)}s`);
          totals.downloaded += 1;
        } else if (result.status === 'expired') {
          console.log(`gone (${result.reason})`);
          totals.expired += 1;
        } else {
          console.log(`FAILED (${result.reason})`);
          totals.failed += 1;
        }
      }
    } catch (err) {
      console.log(`  ERROR: ${err.message}`);
      problems.push(`${account.name}: ${err.message.split('\n')[0]}`);
    } finally {
      if (session) session.close();
    }

    if (index < targets.length - 1) await sleep(PAUSE_BETWEEN_ACCOUNTS_MS);
  }

  if (problems.length) {
    console.log('\nAccounts that did not complete:');
    for (const p of problems) console.log(`  ${p}`);
  }

  return totals;
}

// ─── sidecar repair ──────────────────────────────────────────────────────────

// pr1maly_[map_]<unix-time>_<match-id>.dem
const DEMO_NAME_RE = /^pr1maly_(?:(.+)_)?(\d{10})_(\d+)\.dem$/;

/**
 * Rewrite sidecars for demos already on disk, without touching Steam.
 *
 * Everything a sidecar needs is recoverable offline: date and match ID from
 * the filename, the owning account from the ledger. Used to repair demos
 * fetched before the map handling was fixed.
 */
function repairSidecars(accounts, outputDir) {
  const state = ledger.load();
  const byName = new Map(accounts.map((a) => [a.name, a]));

  // filename -> owning account label, from the ledger
  const owners = new Map();
  for (const [label, account] of Object.entries(state.accounts || {})) {
    for (const entry of Object.values(account.matches || {})) {
      if (entry.filename) owners.set(entry.filename, label);
    }
  }

  let repaired = 0;
  for (const filename of fs.readdirSync(outputDir)) {
    const parsed = DEMO_NAME_RE.exec(filename);
    if (!parsed) continue;

    const [, rawMap, matchTime, matchId] = parsed;
    const label = owners.get(filename);
    const account = label ? byName.get(label) : null;

    const info = {
      matchId,
      matchTime: Number(matchTime),
      // "unknown_map" was a placeholder written by an earlier version.
      mapName: rawMap && rawMap !== 'unknown_map' ? rawMap : null,
      playerIds: [],
    };

    writeSidecar(path.join(outputDir, filename), info, account ? account.steam_id : null);
    console.log(`  rewrote ${filename}.info${info.mapName ? '' : ' (map left to the demo header)'}`);
    repaired += 1;
  }

  console.log(`\nRepaired ${repaired} sidecar(s).`);
  if (repaired) console.log('Re-run Sync Folder in the app to pick up the corrected metadata.');
}

// ─── entry point ─────────────────────────────────────────────────────────────

function arg(args, flag) {
  const index = args.indexOf(flag);
  return index === -1 ? null : args[index + 1] || null;
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const only = arg(args, '--account');
  const resolverName = arg(args, '--resolver');
  const requestedMode = arg(args, '--mode');

  const { dir: outputDir, source } = resolveOutputDir();
  console.log(`\nDemo folder : ${outputDir}  (${source})`);
  if (dryRun) console.log('Dry run     : nothing will be downloaded');
  if (!dryRun) sweepPartialFiles(outputDir);

  const stored = tokens.loadAll();
  const accounts = loadAccounts();

  if (args.includes('--repair-sidecars')) {
    console.log('Mode        : repairing sidecars (no Steam connection)\n');
    repairSidecars(accounts, outputDir);
    return;
  }

  // Prefer the ledger when it has anything to do — it has no window limit and
  // only the resolver account can collide with a running Steam client.
  const hasLedgerWork = ledger.pendingMatches(ledger.load()).length > 0;
  const mode = requestedMode || (hasLedgerWork ? 'ledger' : 'recent');

  if (mode !== 'ledger' && mode !== 'recent') {
    throw new Error(`Unknown mode "${mode}". Use --mode ledger or --mode recent.`);
  }

  const debug = args.includes('--debug');

  const rawLimit = arg(args, '--limit');
  const limit = rawLimit === null ? 0 : parseInt(rawLimit, 10);
  if (rawLimit !== null && (!Number.isFinite(limit) || limit < 1)) {
    throw new Error(`--limit expects a whole number of 1 or more, got "${rawLimit}".`);
  }

  const totals =
    mode === 'ledger'
      ? await runLedgerMode(accounts, stored, outputDir, { dryRun, resolverName, debug, limit })
      : await runRecentMode(accounts, stored, outputDir, { dryRun, only, limit });

  console.log('\n─────────────────────────────────────────');
  console.log(`Downloaded : ${totals.downloaded}`);
  console.log(`Already had: ${totals.skipped}`);
  console.log(`Unavailable: ${totals.expired}`);
  if (totals.failed) console.log(`Failed     : ${totals.failed}`);

  if (totals.downloaded && !dryRun) {
    console.log('\nNow open the app and use Sync Folder to import them.');
  }
  console.log();
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(`\n${err.message}\n`);
    process.exit(1);
  });
