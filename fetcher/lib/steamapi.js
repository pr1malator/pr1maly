'use strict';

// Walks a Steam account's match history using the public Web API.
//
//   ICSGOPlayers_730/GetNextMatchSharingCode/v1
//
// Given the share code of a match you already know about, this returns the
// share code of the one played after it. Chaining those calls enumerates
// history indefinitely — no client session, no Game Coordinator, and none of
// the fixed-size window that limits requestRecentGames.
//
// The per-account "steamidkey" is the match-sharing authentication code from
// Steam's CS2 game-data page. It does not expire unless regenerated.

const ENDPOINT = 'https://api.steampowered.com/ICSGOPlayers_730/GetNextMatchSharingCode/v1';

// Valve throttles this. One match per call means a backlog walk can be long,
// so pace it rather than risk a temporary block.
const DELAY_BETWEEN_CALLS_MS = 1_000;
const MAX_STEPS_PER_RUN = 500;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Fetch the share code of the match played after `knownCode`.
 * @returns {Promise<string|null>} null when there is no newer match
 */
async function getNextShareCode({ apiKey, steamId, authCode, knownCode }) {
  const url =
    `${ENDPOINT}?key=${encodeURIComponent(apiKey)}` +
    `&steamid=${encodeURIComponent(steamId)}` +
    `&steamidkey=${encodeURIComponent(authCode)}` +
    `&knowncode=${encodeURIComponent(knownCode)}`;

  const response = await fetch(url);

  // Valve signals "nothing newer" with 202 and an empty body.
  if (response.status === 202 || response.status === 204) return null;

  if (response.status === 403) {
    throw new Error(
      'Steam rejected the request (403). Either the Web API key is wrong, or the ' +
        "match-sharing auth code does not belong to this account's Steam ID."
    );
  }

  if (response.status === 412) {
    throw new Error(
      'Steam rejected the known share code (412). The stored code is not a valid ' +
        'match for this account — re-seed it from the CS2 game-data page.'
    );
  }

  if (response.status === 429) {
    throw new Error('Rate limited by Steam (429). Wait a few minutes and run again.');
  }

  if (!response.ok) {
    throw new Error(`Steam Web API returned HTTP ${response.status}.`);
  }

  let body;
  try {
    body = await response.json();
  } catch {
    return null; // empty body also means "nothing newer"
  }

  const next = body && body.result && body.result.nextcode;
  if (!next || next === 'n/a') return null;
  return next;
}

/**
 * Walk forward from `fromCode` until Steam reports no newer match.
 *
 * @param {function(string): void} onFound called with each new share code, in
 *        play order, so the caller can persist progress as it goes
 * @returns {Promise<{codes: string[], cursor: string, stoppedEarly: boolean}>}
 */
async function walkHistory({ apiKey, steamId, authCode, fromCode, onFound }) {
  const codes = [];
  let cursor = fromCode;
  let stoppedEarly = false;

  for (let step = 0; step < MAX_STEPS_PER_RUN; step += 1) {
    const next = await getNextShareCode({ apiKey, steamId, authCode, knownCode: cursor });
    if (!next) return { codes, cursor, stoppedEarly: false };

    codes.push(next);
    cursor = next;
    if (onFound) onFound(next);

    await sleep(DELAY_BETWEEN_CALLS_MS);

    if (step === MAX_STEPS_PER_RUN - 1) stoppedEarly = true;
  }

  return { codes, cursor, stoppedEarly };
}

module.exports = { getNextShareCode, walkHistory, MAX_STEPS_PER_RUN };
