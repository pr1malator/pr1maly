'use strict';

// One-time Steam authentication per account.
//
//   node auth.js <account>          log in with password + Steam Guard
//   node auth.js <account> --qr     log in by scanning a QR code
//   node auth.js --status           show which accounts have a stored token
//   node auth.js --forget <account> delete a stored token
//
// <account> is the "name" field from data/accounts.json.
//
// What gets stored is a refresh token, never your password. fetch.js uses it
// to mint short-lived sessions without prompting again.

const readline = require('readline');

const { LoginSession, EAuthTokenPlatformType, EAuthSessionGuardType } = require('steam-session');
const qrcode = require('qrcode-terminal');

const { loadAccounts } = require('./lib/paths');
const tokens = require('./lib/tokens');

const LOGIN_TIMEOUT_MS = 180_000;

function ask(query, { hidden = false, trim = true } = {}) {
  return new Promise((resolve) => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
    });

    let muted = false;
    rl._writeToOutput = (str) => {
      if (!muted) rl.output.write(str);
    };

    rl.question(query, (answer) => {
      rl.close();
      if (hidden) process.stdout.write('\n');
      // Passwords are never trimmed — leading/trailing spaces are legal.
      resolve(trim ? answer.trim() : answer);
    });

    // question() writes the prompt synchronously, so muting afterwards hides
    // only the typed characters.
    muted = hidden;
  });
}

/** Resolve a CLI account name against data/accounts.json. */
function findAccount(name) {
  const accounts = loadAccounts();
  const match = accounts.find((a) => a.name.toLowerCase() === name.toLowerCase());
  if (match) return match;

  const known = accounts.map((a) => `  ${a.name}  (${a.steam_id})`).join('\n');
  throw new Error(`No account named "${name}" in data/accounts.json.\n\nKnown accounts:\n${known}`);
}

/** Resolve once the session authenticates, or reject on error/timeout. */
function waitForAuth(session) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      session.cancelLoginAttempt();
      reject(new Error('Timed out waiting for the login to be approved.'));
    }, LOGIN_TIMEOUT_MS);

    const done = (fn) => (arg) => {
      clearTimeout(timer);
      fn(arg);
    };

    session.on('authenticated', done(resolve));
    session.on('error', done(reject));
    session.on(
      'timeout',
      done(() => reject(new Error('Steam timed out the login attempt.')))
    );
  });
}

async function loginWithCredentials(account, loginNameFromArgs) {
  // The "name" in accounts.json is a label you chose in the app — it is not
  // necessarily your Steam login name. Ask, defaulting to the label.
  const stored = tokens.get(account.name);
  const suggestion = loginNameFromArgs || (stored && stored.accountName) || account.name;

  const entered = await ask(`Steam login name [${suggestion}]: `);
  const accountName = entered || suggestion;

  const password = await ask(`Steam password for ${accountName}: `, {
    hidden: true,
    trim: false,
  });
  if (!password) throw new Error('No password entered.');

  const session = new LoginSession(EAuthTokenPlatformType.SteamClient);
  const started = await session.startWithCredentials({ accountName, password });

  if (started.actionRequired) {
    const actions = started.validActions || [];
    const codeAction = actions.find(
      (a) =>
        a.type === EAuthSessionGuardType.EmailCode || a.type === EAuthSessionGuardType.DeviceCode
    );

    if (codeAction) {
      const where =
        codeAction.type === EAuthSessionGuardType.EmailCode
          ? `the email sent to ${codeAction.detail || 'your address'}`
          : 'your Steam mobile authenticator';
      const code = await ask(`Steam Guard code from ${where}: `);
      await session.submitSteamGuardCode(code);
    } else {
      console.log('\nApprove this login in your Steam mobile app, then wait...');
    }
  }

  await waitForAuth(session);
  return session;
}

async function loginWithQR() {
  // No username and no password involved — Steam identifies the account from
  // whichever one scans the code.
  const session = new LoginSession(EAuthTokenPlatformType.SteamClient);
  const started = await session.startWithQR();

  console.log('\nScan this with the Steam mobile app (Steam Guard -> scan QR):\n');
  qrcode.generate(started.qrChallengeUrl, { small: true });
  console.log(`\nIf the code will not scan, open this URL on the phone instead:\n${started.qrChallengeUrl}\n`);

  session.on('remoteInteraction', () => {
    console.log('Scanned. Now approve it in the app...');
  });

  await waitForAuth(session);
  return session;
}

function showStatus() {
  const accounts = loadAccounts();
  const stored = tokens.loadAll();

  if (!accounts.length) {
    console.log('No accounts with a Steam ID in data/accounts.json.');
    return;
  }

  console.log(`\nToken store: ${tokens.TOKENS_FILE}\n`);
  for (const account of accounts) {
    const entry = stored[account.name];
    const state = entry ? `authenticated ${entry.savedAt.slice(0, 10)}` : 'NOT authenticated';
    console.log(`  ${account.name.padEnd(16)} ${String(account.steam_id).padEnd(20)} ${state}`);
  }

  const missing = accounts.filter((a) => !stored[a.name]);
  if (missing.length) {
    console.log(`\nTo authenticate: npm run auth -- ${missing[0].name}`);
  }
  console.log();
}

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--status')) {
    showStatus();
    return;
  }

  const forgetIndex = args.indexOf('--forget');
  if (forgetIndex !== -1) {
    const name = args[forgetIndex + 1];
    if (!name) throw new Error('Usage: node auth.js --forget <account>');
    console.log(tokens.remove(name) ? `Removed stored token for ${name}.` : `No stored token for ${name}.`);
    return;
  }

  const useQR = args.includes('--qr');

  // --login lets you give the Steam login name up front when it differs from
  // the label in accounts.json.
  const loginIndex = args.indexOf('--login');
  const loginName = loginIndex !== -1 ? args[loginIndex + 1] : null;

  const positional = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--login');
  const name = positional[0];
  if (!name) {
    console.log(
      'Usage: node auth.js <account> [--qr] [--login <steam-login-name>]\n' +
        '       node auth.js --status'
    );
    showStatus();
    return;
  }

  const account = findAccount(name);
  console.log(`\nAuthenticating "${account.name}" (${account.steam_id})`);

  const session = useQR
    ? await loginWithQR()
    : await loginWithCredentials(account, loginName);
  const loggedInAs = session.steamID ? session.steamID.getSteamID64() : null;

  if (loggedInAs && loggedInAs !== String(account.steam_id)) {
    console.warn(
      `\nWARNING: you logged in as ${loggedInAs}, but data/accounts.json lists ` +
        `${account.steam_id} for "${account.name}".\n` +
        'Demos will be fetched for the account you actually logged into.'
    );
  }

  tokens.save(account.name, {
    refreshToken: session.refreshToken,
    steamId: loggedInAs,
    accountName: session.accountName || null,
  });
  session.cancelLoginAttempt();

  console.log(`\nDone. Refresh token stored for "${account.name}".`);
  console.log('Run "npm start" to fetch demos.\n');
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(`\n${err.message}\n`);
    process.exit(1);
  });
