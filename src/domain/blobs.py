"""Reading the opaque JSON columns back off a stored match.

``aim_stats``, ``role_data``, ``utility_data``, ``impact_stats``,
``enriched_json`` and ``replay_json`` are TEXT. Nothing in the schema describes
them, so every consumer has been writing its own ``try: json.loads(...) except:
...`` — a dozen copies with four different except clauses between them, some
returning ``None``, some ``{}``, some letting a TypeError through.

That matters more here than the duplication suggests. These blobs are the only
surviving record of a match whose demo the retention feature has deleted, and
they were written by whatever analyzer version was current at the time. A row
from a year ago is a shape nobody is thinking about, and a decoder that raises
on it takes down a page rather than degrading.

So: one decoder, tolerant by construction, with the caller saying what an
absent or unreadable value should look like.
"""

from __future__ import annotations

import json
from typing import Any

# The blob columns on `matches`. round_stats carries enriched_json and
# replay_json, which are decoded by the readers in src/database.py.
MATCH_BLOB_COLUMNS = ("aim_stats", "role_data", "utility_data", "impact_stats")


def decode(raw: Any, default: Any = None) -> Any:
    """Decode one stored blob. Never raises.

    Accepts what the database actually hands back, which is a string, or None
    for a match analysed before that measurement existed — and also an
    already-decoded value, because some callers decode before passing it on.
    """
    if raw is None or raw == "":
        return default
    if not isinstance(raw, str | bytes | bytearray):
        return raw  # already decoded
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return default
    return default if decoded is None else decoded


def decode_dict(raw: Any) -> dict[str, Any]:
    """Decode a blob expected to be an object, or ``{}``.

    Guards the shape as well as the parse: a blob holding a bare list would
    otherwise reach code that immediately calls ``.get`` on it.
    """
    decoded = decode(raw, default={})
    return decoded if isinstance(decoded, dict) else {}


def decode_match_blobs(match: dict[str, Any]) -> dict[str, Any]:
    """Return *match* with its blob columns decoded in place.

    The single-match view needs all four, and did this four times over with
    four copies of the same try/except.
    """
    for column in MATCH_BLOB_COLUMNS:
        if column in match:
            match[column] = decode(match[column])
    return match


def stored_value(match: dict[str, Any], column: str, path: str) -> Any:
    """One field out of one blob, e.g. ``stored_value(m, "aim_stats", "aim_rating")``.

    For the trend and career pages, which want a single headline number out of
    each blob and should not care that it arrived as text.
    """
    value: Any = decode_dict(match.get(column))
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
