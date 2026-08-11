'use strict';

// Persistence for Steam refresh tokens.
//
// A refresh token grants full account access for its entire lifetime, so this
// file is password-equivalent. It lives in data/, which .gitignore already
// excludes, and is written with owner-only permissions where the platform
// supports them.

const fs = require('fs');
const path = require('path');

const { TOKENS_FILE } = require('./paths');

function loadAll() {
  try {
    return JSON.parse(fs.readFileSync(TOKENS_FILE, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return {};
    throw new Error(`Could not read ${TOKENS_FILE}: ${err.message}`);
  }
}

function get(accountName) {
  return loadAll()[accountName] || null;
}

/**
 * @param {string} label        the account's "name" in data/accounts.json
 * @param {object} entry
 * @param {string} entry.accountName  the real Steam login name, if known
 */
function save(label, { refreshToken, steamId, accountName }) {
  const all = loadAll();
  all[label] = {
    refreshToken,
    steamId: steamId ? String(steamId) : null,
    accountName: accountName || null,
    savedAt: new Date().toISOString(),
  };

  fs.mkdirSync(path.dirname(TOKENS_FILE), { recursive: true });
  fs.writeFileSync(TOKENS_FILE, JSON.stringify(all, null, 2) + '\n', {
    encoding: 'utf8',
    mode: 0o600,
  });

  // writeFileSync only applies mode when creating the file, so enforce it on
  // rewrites too. Silently ignored on filesystems without POSIX permissions.
  try {
    fs.chmodSync(TOKENS_FILE, 0o600);
  } catch {
    /* not supported on this platform */
  }
}

function remove(accountName) {
  const all = loadAll();
  if (!(accountName in all)) return false;
  delete all[accountName];
  fs.writeFileSync(TOKENS_FILE, JSON.stringify(all, null, 2) + '\n', 'utf8');
  return true;
}

module.exports = { loadAll, get, save, remove, TOKENS_FILE };
