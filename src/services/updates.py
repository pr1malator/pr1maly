"""Is there a newer release? Asked only when the user asks for it.

The app is self-hosted and the README promises no external services are
required. An update check is a request to a third party, so it is built to
keep that promise literally true:

  - Off by default. Nothing is contacted until someone turns it on, or presses
    the button, which is the same standing as the AI providers and the Steam
    fetcher.
  - The request carries nothing. It fetches a static file and compares
    versions here. No current version in the query string, no install id, no
    counter — so what reaches the other end is an IP and a timestamp, which is
    unavoidable for any request at all and is said plainly in the UI.
  - What reaches the other end is not written down. The file is served from
    this project's own site with access logging off for that path, so the
    author cannot count installs either. That costs a number worth having and
    buys the only version of "nothing is recorded" that can be checked instead
    of believed.
  - It never installs anything. This process owns the user's match database and
    migrates its schema at startup; an unattended upgrade of that, while nobody
    is watching, is a far worse trade than running a version behind. What comes
    back is a version, a link to the notes, and the command to run.

The answer is cached for a day, so leaving the setting on costs one request per
day and works offline in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from src.config.store import JsonStore
from src.version import APP_VERSION, is_newer

# A static file on the project's own site, not the GitHub API: the API
# rate-limits by IP, returns a great deal more than is needed, and is a moving
# surface. tools/build_release.py writes this from the same version the release
# workflow checks against the git tag.
#
# The project's own site rather than a hosted one, which is a choice about who
# holds the log rather than about convenience. Any host of this file sees the
# unavoidable minimum — an IP, a timestamp, a user agent — and putting it on
# someone else's pages service would hand that to them instead of to us, which
# is not obviously the friendlier arrangement for the person checking. Ours is
# served with logging turned off for this path:
#
#     location = /latest.json { access_log off; }
#
# so the answer to "what happens to my IP" is a line of the server config
# rather than an assurance. Nothing else about the request changes: it carries
# no version, no identifier and no query string, and tests/test_updates.py
# fails if that stops being true.
LATEST_URL = "https://www.pr1maly.com/latest.json"

CACHE_HOURS = 24
TIMEOUT_SECONDS = 5.0

config_store = JsonStore(
    "update_config.json",
    defaults={
        # Off. Turning this on is the user saying the check may reach the
        # network; nothing else in this file assumes permission.
        "enabled": False,
        "url": LATEST_URL,
    },
)

cache_store = JsonStore(
    "update_cache.json",
    defaults={"checked_at": "", "version": "", "released": "", "notes": ""},
)


@dataclass(frozen=True)
class UpdateStatus:
    current: str
    latest: str | None
    update_available: bool
    notes: str | None
    released: str | None
    checked_at: str | None
    enabled: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "latest": self.latest,
            "update_available": self.update_available,
            "notes": self.notes,
            "released": self.released,
            "checked_at": self.checked_at,
            "enabled": self.enabled,
            "error": self.error,
        }


def _cache_is_fresh(checked_at: str) -> bool:
    if not checked_at:
        return False
    try:
        when = datetime.fromisoformat(checked_at)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return datetime.now(UTC) - when < timedelta(hours=CACHE_HOURS)


def _status_from(cache: dict[str, Any], *, enabled: bool, error: str | None = None) -> UpdateStatus:
    latest = cache.get("version") or None
    return UpdateStatus(
        current=APP_VERSION,
        latest=latest,
        update_available=bool(latest) and is_newer(latest, APP_VERSION),
        notes=cache.get("notes") or None,
        released=cache.get("released") or None,
        checked_at=cache.get("checked_at") or None,
        enabled=enabled,
        error=error,
    )


def fetch_latest(url: str | None = None) -> dict[str, Any]:
    """Ask the published file what the newest version is. Raises on failure."""
    config = config_store.read()
    target = url or config.get("url") or LATEST_URL
    # No query string, no custom headers beyond a plain user agent: everything
    # added here would be something the user did not agree to send.
    response = httpx.get(target, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("version"):
        raise ValueError("the published file does not name a version")
    return payload


def check(*, force: bool = False) -> UpdateStatus:
    """Current update status.

    Without *force*, this contacts nothing unless the setting is on and the
    cached answer is a day old. With it — the "check now" button — it asks
    regardless, because that press is the permission.
    """
    config = config_store.read()
    enabled = bool(config.get("enabled"))
    cache = cache_store.read()

    if not force and (not enabled or _cache_is_fresh(str(cache.get("checked_at", "")))):
        return _status_from(cache, enabled=enabled)

    try:
        payload = fetch_latest()
    except Exception as exc:  # offline, DNS, 404, malformed — none are fatal
        return _status_from(cache, enabled=enabled, error=str(exc))

    cache = {
        "checked_at": datetime.now(UTC).isoformat(),
        "version": str(payload.get("version", "")),
        "released": str(payload.get("released", "")),
        "notes": str(payload.get("notes", "")),
    }
    cache_store.write(cache)
    return _status_from(cache, enabled=enabled)


def set_enabled(enabled: bool) -> UpdateStatus:
    config = config_store.read()
    config["enabled"] = bool(enabled)
    config_store.write(config)
    # Turning it off should stop showing an answer obtained under the old
    # setting, so the cached result goes with it.
    if not enabled:
        cache_store.write(cache_store.defaults.copy())
    return check()
