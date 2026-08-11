'use strict';

// Persistent record of every match Valve has told us about.
//
// This is deliberately independent of demo downloading. Walking the ledger
// needs no Steam session at all — only the Web API — so the app can always
// tell the difference between "played nothing that week" and "the fetcher was
// broken". Demos expire; this record does not.
//
// Stored in data/, which .gitignore already excludes. It holds a Steam Web API
// key and per-account match-sharing auth codes, so it is written owner-only.

const fs = require('fs');
const path = require('path');

const { DATA_DIR } = require('./paths');

const LEDGER_FILE = path.join(DATA_DIR, 'steam_sharecodes.json');

/** @typedef {'pending'|'downloaded'|'expired'|'failed'} MatchStatus */

function load() {
  try {
    const parsed = JSON.parse(fs.readFileSync(LEDGER_FILE, 'utf8'));
    return { apiKey: null, accounts: {}, ...parsed };
  } catch (err) {
    if (err.code === 'ENOENT') return { apiKey: null, accounts: {} };
    throw new Error(`Could not read ${LEDGER_FILE}: ${err.message}`);
  }
}

function save(state) {
  fs.mkdirSync(path.dirname(LEDGER_FILE), { recursive: true });
  fs.writeFileSync(LEDGER_FILE, JSON.stringify(state, null, 2) + '\n', {
    encoding: 'utf8',
    mode: 0o600,
  });
  try {
    fs.chmodSync(LEDGER_FILE, 0o600);
  } catch {
    /* not supported on this platform */
  }
}

/** Env var wins so the key can stay out of the file entirely if preferred. */
function getApiKey(state) {
  return process.env.STEAM_API_KEY || state.apiKey || null;
}

function getAccount(state, label) {
  return state.accounts[label] || null;
}

/** Create the account entry if absent, and return it. */
function ensureAccount(state, label) {
  if (!state.accounts[label]) {
    state.accounts[label] = {
      authCode: null,
      seedShareCode: null,
      cursor: null,
      // Tracked and fetched by default; both are opt-out per account.
      walkEnabled: true,
      downloadEnabled: true,
      matches: {},
    };
  }
  return state.accounts[label];
}

/**
 * Whether an account participates in a given stage. Absent means enabled, so
 * ledgers written before these flags existed keep working.
 */
function isEnabled(account, stage) {
  const key = stage === 'walk' ? 'walkEnabled' : 'downloadEnabled';
  return account[key] !== false;
}

/** Accounts that have everything needed to walk the ledger, and are enabled for it. */
function configuredAccounts(state) {
  return Object.entries(state.accounts)
    .filter(([, cfg]) => cfg.authCode && (cfg.cursor || cfg.seedShareCode))
    .filter(([, cfg]) => isEnabled(cfg, 'walk'))
    .map(([label, cfg]) => ({ label, ...cfg }));
}

/**
 * Record a newly discovered share code. Idempotent — re-walking never
 * resets the status of a match already downloaded.
 */
function recordMatch(state, label, shareCode) {
  const account = ensureAccount(state, label);
  if (account.matches[shareCode]) return false;

  account.matches[shareCode] = {
    discoveredAt: new Date().toISOString(),
    status: /** @type {MatchStatus} */ ('pending'),
    filename: null,
  };
  return true;
}

function markMatch(state, label, shareCode, status, extra = {}) {
  const account = ensureAccount(state, label);
  if (!account.matches[shareCode]) recordMatch(state, label, shareCode);

  account.matches[shareCode] = {
    ...account.matches[shareCode],
    ...extra,
    status,
    updatedAt: new Date().toISOString(),
  };
}

/**
 * Every match not yet successfully downloaded, oldest first.
 * "expired" is terminal — Valve deleted the demo, retrying cannot help.
 */
function pendingMatches(state) {
  const pending = [];
  for (const [label, account] of Object.entries(state.accounts)) {
    if (!isEnabled(account, 'download')) continue;

    for (const [shareCode, entry] of Object.entries(account.matches || {})) {
      if (entry.status === 'pending' || entry.status === 'failed') {
        pending.push({ label, shareCode, ...entry });
      }
    }
  }
  return pending.sort((a, b) => String(a.discoveredAt).localeCompare(String(b.discoveredAt)));
}

function counts(account) {
  const tally = { total: 0, pending: 0, downloaded: 0, expired: 0, failed: 0 };
  for (const entry of Object.values(account.matches || {})) {
    tally.total += 1;
    if (tally[entry.status] !== undefined) tally[entry.status] += 1;
  }
  return tally;
}

module.exports = {
  LEDGER_FILE,
  load,
  save,
  getApiKey,
  getAccount,
  ensureAccount,
  isEnabled,
  configuredAccounts,
  recordMatch,
  markMatch,
  pendingMatches,
  counts,
};
