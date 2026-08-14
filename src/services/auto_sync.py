"""The Auto-Sync loop: work through the backlog one match at a time, unattended.

A cycle is deliberately one unit of work — one download, or one import, or one
check — so switching the feature off never has to interrupt more than a single
demo, and a user pressing a button never waits on a whole backlog.

This was module-global state in api.py driving a daemon thread. The policy is
the valuable part and it is now separable from the plumbing: everything the loop
needs the rest of the app to do arrives as :class:`AutoSyncOps`, so the decision
table can be tested without Steam, a disk or a subprocess.

Two rules the code exists to enforce, both learned the hard way:

  Never disable itself. A Steam outage or one unparseable demo must not silently
  end a background job the user expects to still be running when they return.
  Failures back off exponentially to a ceiling; they never stop the loop.

  Never fight the user. Every step goes through the same single job slot, and
  JobBusy simply means "come back shortly".
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from src.services.demo_files import DemoFolderUnusable
from src.services.steam_jobs import FetcherUnavailable, JobBusy, utc_now

# How many activity lines the UI feed keeps.
MAX_ACTIVITY = 40

# Waits, in seconds, for the situations that are not the configured interval.
BUSY_WAIT = 20          # a manual job holds the slot
PLAYING_WAIT = 120      # someone is in a match
BACKOFF_BASE = 60       # first retry after a failed step
BACKOFF_MAX = 1800      # ceiling on the exponential backoff

# Consecutive import failures before a demo is set aside.
SKIP_AFTER = 3

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "interval_minutes": 5,       # gap between finished matches; 0 = no gap
    "idle_check_minutes": 30,    # how often to ask Steam for new matches when idle
    "pause_when_playing": True,  # hold off while a tracked account is in CS2
}


@dataclass(frozen=True)
class AutoSyncOps:
    """What the loop needs the rest of the application to do for it.

    Passed as callables rather than imported, so the policy has no opinion about
    where demos come from and the tests do not need Steam.
    """

    load_config: Callable[[], dict[str, Any]]
    is_playing: Callable[[], dict[str, Any]]
    next_unimported: Callable[[set[str]], str | None]
    import_one: Callable[[str], bool]
    outstanding: Callable[[], int]
    run_job: Callable[[str, list[str]], int]
    last_job_error: Callable[[], str | None]


def backoff(streak: int) -> float:
    """Exponential, capped. Never zero — that would be a hot loop against a
    failing Steam."""
    return min(BACKOFF_BASE * (2 ** max(0, streak - 1)), BACKOFF_MAX)


class AutoSync:
    def __init__(self, ops: AutoSyncOps) -> None:
        self._ops = ops
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._empty_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            # off|waiting|checking|downloading|importing|paused|blocked|error
            "phase": "off",
            "detail": "",
            "next_action_at": None,
            "started_at": None,
            "last_error": None,
            "failure_streak": 0,
            "totals": {"downloaded": 0, "imported": 0, "failed": 0},
            "activity": [],
            # Demos that refused to parse. Without this one corrupt file would
            # be retried forever and no other match would ever be reached.
            "skipped": {},
        }

    # -- observable state --------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(
                self._state,
                totals=dict(self._state["totals"]),
                activity=list(self._state["activity"]),
                skipped=dict(self._state["skipped"]),
            )

    def note(self, text: str) -> None:
        """Record a line in the activity feed shown in the UI."""
        with self._lock:
            self._state["activity"].append({"at": utc_now(), "text": text})
            if len(self._state["activity"]) > MAX_ACTIVITY:
                del self._state["activity"][:-MAX_ACTIVITY]

    def set_phase(self, phase: str, detail: str = "", wait_seconds: float | None = None) -> None:
        next_at = None
        if wait_seconds is not None:
            next_at = (datetime.now(UTC) + timedelta(seconds=wait_seconds)).isoformat()
        with self._lock:
            self._state["phase"] = phase
            self._state["detail"] = detail
            self._state["next_action_at"] = next_at

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def set_aside(self) -> set[str]:
        with self._lock:
            return {
                name for name, attempts in self._state["skipped"].items()
                if attempts >= SKIP_AFTER
            }

    # -- one cycle ---------------------------------------------------------

    def step(self, cfg: dict[str, Any]) -> float:
        """Do one unit of work. Returns how long to wait before the next one.

        The configured interval is only applied after a match is fully imported;
        the download and the import of the same match run back to back.
        """
        interval = max(0, int(cfg["interval_minutes"])) * 60

        # Someone is playing. Downloading 280 MB mid-match costs them ping, and
        # the fetcher's own Game Coordinator login can drop them out of the game.
        if cfg["pause_when_playing"]:
            presence = self._ops.is_playing()
            if presence["playing"]:
                self.set_phase("paused", presence["detail"], PLAYING_WAIT)
                return PLAYING_WAIT

        # A demo already on disk but not yet analysed is the cheapest useful
        # work, and clears the way before another 280 MB lands. It also
        # validates the demo folder before anything is downloaded into it.
        try:
            pending = self._ops.next_unimported(self.set_aside)
        except DemoFolderUnusable as exc:
            self.set_phase("error", str(exc), BACKOFF_MAX)
            self._record(last_error=str(exc))
            return BACKOFF_MAX

        if pending:
            return self._import(pending, interval)

        outstanding = self._ops.outstanding()
        if outstanding > 0:
            return self._download(outstanding, interval)

        return self._check(cfg)

    def _import(self, filename: str, interval: float) -> float:
        self.set_phase("importing", filename)
        try:
            ok = self._ops.import_one(filename)
        except JobBusy:
            self.set_phase("blocked", "waiting for the job you started", BUSY_WAIT)
            return BUSY_WAIT

        with self._lock:
            totals = self._state["totals"]
            totals["imported" if ok else "failed"] += 1
            self._state["failure_streak"] = 0 if ok else self._state["failure_streak"] + 1
            streak = self._state["failure_streak"]
            if ok:
                self._state["skipped"].pop(filename, None)
                attempts = 0
            else:
                attempts = self._state["skipped"].get(filename, 0) + 1
                self._state["skipped"][filename] = attempts
                self._state["last_error"] = f"Could not import {filename}"

        # A demo that will not parse is set aside after a few tries. Retrying it
        # forever would block every other match behind one corrupt file, and
        # nothing about a re-read is going to make it parse.
        if attempts >= SKIP_AFTER:
            self.note(f"Set aside {filename} after {attempts} failed attempts")
            self.set_phase("waiting", "skipping a demo that will not parse", 5)
            return 5

        wait = interval if ok else backoff(streak)
        self.set_phase("waiting", "next match" if ok else "retrying after a failure", wait)
        return wait

    def _download(self, outstanding: int, interval: float) -> float:
        self.set_phase("downloading", f"1 of {outstanding} outstanding")
        try:
            code = self._ops.run_job("download", ["fetch.js", "--limit", "1"])
        except JobBusy as exc:
            self.set_phase("blocked", str(exc), BUSY_WAIT)
            return BUSY_WAIT
        except FetcherUnavailable as exc:
            self.set_phase("error", str(exc), BACKOFF_MAX)
            return BACKOFF_MAX

        if self._stop.is_set():
            return 0

        if code == 0 and self._ops.next_unimported(self.set_aside):
            with self._lock:
                self._state["totals"]["downloaded"] += 1
                self._state["failure_streak"] = 0
                self._state["last_error"] = None
            self.note("Downloaded a demo")
            return 0  # straight on to the import, no gap within one match

        # Exit 0 with nothing new means the match was expired or already held —
        # progress in the ledger, but no demo. Move on without a long backoff.
        if code == 0:
            wait = min(interval, 60) or 5
            self.set_phase("waiting", "nothing new from that match", wait)
            return wait

        problem = self._ops.last_job_error() or f"fetch.js exited {code}"
        with self._lock:
            self._state["failure_streak"] += 1
            self._state["last_error"] = problem
            self._state["totals"]["failed"] += 1
            streak = self._state["failure_streak"]
        wait = backoff(streak)
        self.set_phase("error", problem, wait)
        return wait

    def _check(self, cfg: dict[str, Any]) -> float:
        self.set_phase("checking", "asking Steam for new matches")
        try:
            code = self._ops.run_job("check", ["sharecodes.js", "--walk"])
        except JobBusy as exc:
            self.set_phase("blocked", str(exc), BUSY_WAIT)
            return BUSY_WAIT
        except FetcherUnavailable as exc:
            self.set_phase("error", str(exc), BACKOFF_MAX)
            return BACKOFF_MAX

        if self._stop.is_set():
            return 0

        outstanding = self._ops.outstanding()
        if code == 0 and outstanding > 0:
            self.note(f"Found {outstanding} match(es) to fetch")
            self._record(failure_streak=0, last_error=None)
            return 0  # start on it immediately

        if code != 0:
            problem = self._ops.last_job_error() or f"sharecodes.js exited {code}"
            with self._lock:
                self._state["failure_streak"] += 1
                self._state["last_error"] = problem
                streak = self._state["failure_streak"]
            wait = backoff(streak)
            self.set_phase("error", problem, wait)
            return wait

        idle = max(1, int(cfg["idle_check_minutes"])) * 60
        self.set_phase("waiting", "up to date — will check again", idle)
        return idle

    def _record(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    # -- the thread --------------------------------------------------------

    def run_forever(self) -> None:
        self.note("Auto-Sync started")
        while not self._stop.is_set():
            cfg = self._ops.load_config()
            if not cfg["enabled"]:
                break
            try:
                wait = self.step(cfg)
            except Exception as exc:  # never let one bad cycle end the loop
                with self._lock:
                    self._state["failure_streak"] += 1
                    self._state["last_error"] = str(exc)
                    streak = self._state["failure_streak"]
                wait = backoff(streak)
                self.set_phase("error", str(exc), wait)
            if wait > 0:
                self._stop.wait(wait)
        self.set_phase("off", "")
        self.note("Auto-Sync stopped")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._state["started_at"] = utc_now()
            self._state["failure_streak"] = 0
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Signal the loop to end. The caller stops any child process."""
        self._stop.set()
        self.set_phase("off", "")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
