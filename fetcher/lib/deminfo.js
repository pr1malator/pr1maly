'use strict';

// Synthesises a .dem.info sidecar for a downloaded demo.
//
// CS2 writes one of these next to every demo it downloads in-game, and
// src/parser.py:parse_info_file reads three things out of it:
//
//   field 2         varint   match timestamp (unix seconds)
//   field 3 -> 2    string   map name
//   field 5 -> 2 -> repeated field 1, varint   player account IDs
//
// That parser is a lenient hand-rolled protobuf scanner: it skips fields it
// does not recognise, so emitting only those three is enough for the Sync
// Folder scan to pick up the map, the date, and which of your accounts played.

const STEAMID64_BASE = 76561197960265728n;

/** Encode an unsigned integer as a protobuf varint. */
function varint(value) {
  let v = BigInt(value);
  if (v < 0n) throw new RangeError('varint cannot encode a negative value');

  const bytes = [];
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v > 0n) byte |= 0x80;
    bytes.push(byte);
  } while (v > 0n);

  return Buffer.from(bytes);
}

/** Tag byte(s) for a field number and wire type. */
function tag(field, wireType) {
  return varint((field << 3) | wireType);
}

/** A length-delimited (wire type 2) field carrying an arbitrary payload. */
function lengthDelimited(field, payload) {
  return Buffer.concat([tag(field, 2), varint(payload.length), payload]);
}

/** A varint (wire type 0) field. */
function varintField(field, value) {
  return Buffer.concat([tag(field, 0), varint(value)]);
}

/** Convert a 64-bit Steam ID to the 32-bit account ID stored in the sidecar. */
function toAccountId(steamId64) {
  const id = BigInt(steamId64);
  return id > STEAMID64_BASE ? id - STEAMID64_BASE : id;
}

/**
 * Build the bytes of a .dem.info sidecar.
 *
 * @param {object}   match
 * @param {number}   match.matchTime  unix seconds
 * @param {string}   match.mapName    e.g. "de_mirage"
 * @param {string[]} match.steamIds   64-bit Steam IDs of the players
 * @returns {Buffer}
 */
function buildDemInfo({ matchTime, mapName, steamIds = [] }) {
  const parts = [];

  if (matchTime) {
    parts.push(varintField(2, Math.floor(matchTime)));
  }

  if (mapName) {
    // watchablematchinfo { game_map = <map> }
    parts.push(lengthDelimited(3, lengthDelimited(2, Buffer.from(mapName, 'utf8'))));
  }

  const accountIds = steamIds
    .filter(Boolean)
    .map(toAccountId)
    .filter((id) => id > 0n);

  if (accountIds.length) {
    // roundstats { reservation { account_ids = [...] } }
    const reservation = Buffer.concat(accountIds.map((id) => varintField(1, id)));
    parts.push(lengthDelimited(5, lengthDelimited(2, reservation)));
  }

  return Buffer.concat(parts);
}

module.exports = { buildDemInfo, toAccountId };
