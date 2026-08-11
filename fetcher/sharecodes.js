'use strict';

// Configures and updates the match ledger.
//
//   node sharecodes.js --api-key <key>
//   node sharecodes.js --set <account> --auth-code <code> --share-code <code>
//   node sharecodes.js --walk [--account <name>]
//   node sharecodes.js --status
//
// Walking needs no Steam session — it only calls the public Web API — so it
// works while Steam is running, while you are in a match, at any time.

const { loadAccounts } = require('./lib/paths');
const ledger = require('./lib/ledger');
const { walkHistory, MAX_STEPS_PER_RUN } = require('./lib/steamapi');

const SHARE_CODE_RE = /^CSGO(-[ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789]{5}){5}$/;
const AUTH_CODE_RE = /^[A-Za-z0-9]{4}-[A-Za-z0-9]{5}-[A-Za-z0-9]{4}$/;

const HELP = `
Match ledger — enumerates your CS2 match history without a Steam session.

  node sharecodes.js --api-key <key>
      Store your Steam Web API key. One key covers every account: it identifies
      the caller, not the account being queried. Get one at
      https://steamcommunity.com/dev/apikey (needs a non-limited account).
      Alternatively set the STEAM_API_KEY environment variable.

  node sharecodes.js --set <account> --auth-code <code> --share-code <code>
      Store an account's match-sharing auth code and a starting share code.
      Both come from Steam's CS2 game-data page, visited in a browser while
      signed in as that account:
        https://help.steampowered.com/en/wizard/HelpWithGameIssue/?appid=730&issueid=128
      The auth code does not expire. The share code is only a starting point —
      the ledger advances past it on its own.

  node sharecodes.js --walk [--account <name>]
      Ask Steam for every match played since the stored cursor.

  node sharecodes.js --toggle <account> [--walk on|off] [--download on|off]
      Choose which accounts are tracked and which have demos downloaded.
      An account can be tracked without being downloaded — its matches stay
      in the ledger, they are simply not fetched.

  node sharecodes.js --status
      Show what is configured and what the ledger knows.
`;

function arg(args, flag) {
  const index = args.indexOf(flag);
  return index === -1 ? null : args[index + 1] || null;
}

function resolveLabel(name) {
  const accounts = loadAccounts();
  const match = accounts.find((a) => a.name.toLowerCase() === name.toLowerCase());
  if (!match) {
    const known = accounts.map((a) => `  ${a.name}`).join('\n');
    throw new Error(`No account named "${name}" in data/accounts.json.\n\nKnown accounts:\n${known}`);
  }
  return match;
}

function setApiKey(key) {
  const state = ledger.load();
  state.apiKey = key;
  ledger.save(state);
  console.log(`Stored Web API key in ${ledger.LEDGER_FILE}`);
}

function setAccountCodes(name, authCode, shareCode) {
  if (!authCode || !shareCode) {
    throw new Error('Both --auth-code and --share-code are required.');
  }
  if (!AUTH_CODE_RE.test(authCode)) {
    throw new Error(`"${authCode}" does not look like an auth code (expected XXXX-XXXXX-XXXX).`);
  }
  if (!SHARE_CODE_RE.test(shareCode)) {
    throw new Error(
      `"${shareCode}" does not look like a share code ` +
        '(expected CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx).'
    );
  }

  const account = resolveLabel(name);
  const state = ledger.load();
  const entry = ledger.ensureAccount(state, account.name);

  entry.authCode = authCode;
  entry.seedShareCode = shareCode;
  if (!entry.cursor) entry.cursor = shareCode;

  // The seed is itself a real match — record it so it gets downloaded too.
  ledger.recordMatch(state, account.name, shareCode);
  ledger.save(state);

  console.log(`Configured "${account.name}". Now run: node sharecodes.js --walk`);
}

function parseOnOff(value, flag) {
  if (value === null || value === undefined) return null;
  const v = String(value).toLowerCase();
  if (['on', 'yes', 'true', '1'].includes(v)) return true;
  if (['off', 'no', 'false', '0'].includes(v)) return false;
  throw new Error(`${flag} expects "on" or "off", got "${value}".`);
}

function toggleAccount(name, walkValue, downloadValue) {
  const walkFlag = parseOnOff(walkValue, '--walk');
  const downloadFlag = parseOnOff(downloadValue, '--download');
  if (walkFlag === null && downloadFlag === null) {
    throw new Error('Pass at least one of --walk on|off or --download on|off.');
  }

  const account = resolveLabel(name);
  const state = ledger.load();
  const entry = ledger.ensureAccount(state, account.name);

  if (walkFlag !== null) entry.walkEnabled = walkFlag;
  if (downloadFlag !== null) entry.downloadEnabled = downloadFlag;
  ledger.save(state);

  console.log(
    `${account.name}: tracking ${entry.walkEnabled !== false ? 'on' : 'off'}, ` +
      `downloading ${entry.downloadEnabled !== false ? 'on' : 'off'}`
  );
}

async function walk(onlyName) {
  const state = ledger.load();
  const apiKey = ledger.getApiKey(state);
  if (!apiKey) {
    throw new Error('No Steam Web API key set. Run: node sharecodes.js --api-key <key>');
  }

  const accountsByName = new Map(loadAccounts().map((a) => [a.name, a]));
  let configured = ledger.configuredAccounts(state);

  if (onlyName) {
    const target = resolveLabel(onlyName);
    configured = configured.filter((c) => c.label === target.name);
    if (!configured.length) {
      throw new Error(`"${target.name}" has no auth code stored. Run --set for it first.`);
    }
  }

  if (!configured.length) {
    throw new Error('No accounts configured yet. See: node sharecodes.js --help');
  }

  for (const entry of configured) {
    const account = accountsByName.get(entry.label);
    if (!account) {
      console.log(`  ${entry.label}: no longer in accounts.json, skipping`);
      continue;
    }

    process.stdout.write(`${entry.label}: walking history... `);
    let found = 0;

    try {
      const result = await walkHistory({
        apiKey,
        steamId: account.steam_id,
        authCode: entry.authCode,
        fromCode: entry.cursor || entry.seedShareCode,
        onFound: (code) => {
          // Persist as we go so an interruption never loses discovered matches.
          const fresh = ledger.load();
          ledger.recordMatch(fresh, entry.label, code);
          ledger.ensureAccount(fresh, entry.label).cursor = code;
          ledger.save(fresh);
          found += 1;
        },
      });

      console.log(`${found} new match(es).`);
      if (result.stoppedEarly) {
        console.log(
          `  Stopped at the ${MAX_STEPS_PER_RUN}-match safety limit — run --walk again to continue.`
        );
      }
    } catch (err) {
      console.log(`FAILED\n  ${err.message}`);
    }
  }

  console.log('\nRun "npm start" to download demos for anything still pending.');
}

function status() {
  const state = ledger.load();
  const apiKey = ledger.getApiKey(state);
  const accounts = loadAccounts();

  console.log(`\nLedger : ${ledger.LEDGER_FILE}`);
  console.log(`API key: ${apiKey ? 'set' : 'NOT SET — run --api-key <key>'}\n`);

  for (const account of accounts) {
    const entry = ledger.getAccount(state, account.name);
    if (!entry || !entry.authCode) {
      console.log(`  ${account.name.padEnd(16)} not configured`);
      continue;
    }

    const tally = ledger.counts(entry);
    const flags = [
      ledger.isEnabled(entry, 'walk') ? null : 'tracking off',
      ledger.isEnabled(entry, 'download') ? null : 'downloading off',
    ].filter(Boolean);

    console.log(
      `  ${account.name.padEnd(16)} ${String(tally.total).padStart(4)} known  ` +
        `${String(tally.downloaded).padStart(4)} downloaded  ` +
        `${String(tally.pending).padStart(4)} pending  ` +
        `${String(tally.expired).padStart(4)} expired` +
        (tally.failed ? `  ${tally.failed} failed` : '') +
        (flags.length ? `   [${flags.join(', ')}]` : '')
    );
  }
  console.log();
}

async function main() {
  const args = process.argv.slice(2);

  if (!args.length || args.includes('--help') || args.includes('-h')) {
    console.log(HELP);
    return;
  }

  if (args.includes('--status')) return status();

  const apiKey = arg(args, '--api-key');
  if (apiKey) return setApiKey(apiKey);

  const setName = arg(args, '--set');
  if (setName) {
    return setAccountCodes(setName, arg(args, '--auth-code'), arg(args, '--share-code'));
  }

  const toggleName = arg(args, '--toggle');
  if (toggleName) {
    return toggleAccount(toggleName, arg(args, '--walk'), arg(args, '--download'));
  }

  if (args.includes('--walk')) return walk(arg(args, '--account'));

  console.log(HELP);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(`\n${err.message}\n`);
    process.exit(1);
  });
