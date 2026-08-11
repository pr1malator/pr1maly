'use strict';

// Locates the pr1maly data directory and the folder demos should be written to.
// The fetcher is a sibling of api.py, so the repo root is one level up.

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(REPO_ROOT, 'data');

const ACCOUNTS_FILE = path.join(DATA_DIR, 'accounts.json');
const SYNC_CONFIG_FILE = path.join(DATA_DIR, 'sync_config.json');
const TOKENS_FILE = path.join(DATA_DIR, 'steam_tokens.json');

// Where CS2 puts demos you download in-game. Used as a last resort so a fresh
// checkout works without configuration on a default Steam install.
const DEFAULT_REPLAY_DIRS = [
  'C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays',
  path.join(
    process.env.HOME || '',
    '.steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays'
  ),
];

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return fallback;
    throw new Error(`Could not read ${file}: ${err.message}`);
  }
}

/** Accounts from data/accounts.json that have a Steam ID set. */
function loadAccounts() {
  const cfg = readJson(ACCOUNTS_FILE, { accounts: [] });
  return (cfg.accounts || []).filter((a) => a.steam_id && String(a.steam_id).trim());
}

/**
 * Resolve the directory demos are written into, in priority order:
 *   1. DEMO_DIR environment variable
 *   2. the folder in data/sync_config.json, if it exists on this machine
 *   3. a default CS2 replays path, if it exists
 *
 * sync_config.json holds the *container* path (/demos) when running under
 * Docker, which is why it is only used when it actually resolves on disk.
 */
function resolveOutputDir() {
  const candidates = [];

  if (process.env.DEMO_DIR) {
    candidates.push({ dir: process.env.DEMO_DIR, source: 'DEMO_DIR env var' });
  }

  const syncFolder = readJson(SYNC_CONFIG_FILE, {}).folder;
  if (syncFolder) {
    candidates.push({ dir: syncFolder, source: 'data/sync_config.json' });
  }

  for (const dir of DEFAULT_REPLAY_DIRS) {
    if (dir) candidates.push({ dir, source: 'default CS2 replays folder' });
  }

  for (const candidate of candidates) {
    try {
      if (fs.statSync(candidate.dir).isDirectory()) return candidate;
    } catch {
      // Not present on this machine — try the next candidate.
    }
  }

  const tried = candidates.map((c) => `  ${c.dir}  (${c.source})`).join('\n');
  throw new Error(
    `No usable demo output folder found. Tried:\n${tried}\n\n` +
      'Set one explicitly, e.g.:\n' +
      '  set DEMO_DIR=C:\\path\\to\\replays   (cmd)\n' +
      '  $env:DEMO_DIR="C:\\path\\to\\replays" (PowerShell)'
  );
}

module.exports = {
  REPO_ROOT,
  DATA_DIR,
  ACCOUNTS_FILE,
  TOKENS_FILE,
  loadAccounts,
  resolveOutputDir,
};
