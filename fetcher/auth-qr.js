'use strict';

// QR sign-in, driven by the web UI.
//
//   node auth-qr.js <account>
//
// Same flow as `auth.js --qr`, but instead of drawing a QR code in the
// terminal it emits machine-readable events on stdout. api.py picks those up
// and the browser renders the QR image.
//
// Nothing secret is typed or transmitted here: a QR login is a device
// authorisation, so there is no password to put in a web form. That is why
// this one is safe to expose in the UI while the credential path is not.

const QRCode = require('qrcode');
const { LoginSession, EAuthTokenPlatformType } = require('steam-session');

const { loadAccounts } = require('./lib/paths');
const tokens = require('./lib/tokens');

// Lines with this prefix are structured events; everything else is plain
// progress text that ends up in the log panel.
const EVENT_PREFIX = 'STEAM_EVENT ';
const LOGIN_TIMEOUT_MS = 180_000;

function emit(event, payload = {}) {
  process.stdout.write(EVENT_PREFIX + JSON.stringify({ event, ...payload }) + '\n');
}

function findAccount(name) {
  const accounts = loadAccounts();
  const match = accounts.find((a) => a.name.toLowerCase() === name.toLowerCase());
  if (match) return match;

  const known = accounts.map((a) => a.name).join(', ');
  throw new Error(`No account named "${name}" in data/accounts.json. Known: ${known}`);
}

/** Resolve when Steam confirms the login, reject on error or timeout. */
function waitForAuth(session) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      session.cancelLoginAttempt();
      reject(new Error('Timed out waiting for the QR code to be approved.'));
    }, LOGIN_TIMEOUT_MS);

    const settle = (fn) => (arg) => {
      clearTimeout(timer);
      fn(arg);
    };

    session.on('authenticated', settle(resolve));
    session.on('error', settle(reject));
    session.on(
      'timeout',
      settle(() => reject(new Error('Steam timed out the login attempt.')))
    );
  });
}

async function main() {
  const name = process.argv[2];
  if (!name) throw new Error('Usage: node auth-qr.js <account>');

  const account = findAccount(name);
  console.log(`Starting QR sign-in for "${account.name}"...`);

  const session = new LoginSession(EAuthTokenPlatformType.SteamClient);
  const started = await session.startWithQR();

  // Rendered server-side so the page needs no QR library and works offline.
  const svg = await QRCode.toString(started.qrChallengeUrl, {
    type: 'svg',
    margin: 1,
    width: 240,
    color: { dark: '#000000', light: '#ffffff' },
  });

  emit('challenge', { account: account.name, url: started.qrChallengeUrl, svg });
  console.log('Scan the code with the Steam mobile app.');

  session.on('remoteInteraction', () => {
    emit('scanned');
    console.log('Scanned. Approve it in the app...');
  });

  await waitForAuth(session);

  const steamId = session.steamID ? session.steamID.getSteamID64() : null;
  const warning =
    steamId && steamId !== String(account.steam_id)
      ? `Signed in as ${steamId}, but accounts.json lists ${account.steam_id} for "${account.name}". ` +
        'Demos will be fetched for the account actually signed in.'
      : null;

  tokens.save(account.name, {
    refreshToken: session.refreshToken,
    steamId,
    accountName: session.accountName || null,
  });
  session.cancelLoginAttempt();

  emit('authenticated', { account: account.name, steamId, warning });
  console.log(warning || `Signed in as ${steamId}.`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    emit('error', { message: err.message });
    console.error(err.message);
    process.exit(1);
  });
