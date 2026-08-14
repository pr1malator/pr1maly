"""Tests for the Auto-Sync scheduling policy.

`AutoSync.step` decides what one cycle of the background loop does and how long
to wait afterwards. That decision table is the whole behaviour of the feature
and had no tests: it downloads hundreds of megabytes, defers to the user, and is
expected to survive a Steam outage without switching itself off.

Written against the module globals before the class existed, then pointed at it
— the assertions did not change, which is the evidence the move preserved
behaviour. Nothing here touches Steam, the disk or a subprocess: the loop's
dependencies arrive as AutoSyncOps and every one of them is a fake.
"""

from __future__ import annotations

import pytest

from src.services.auto_sync import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    BUSY_WAIT,
    MAX_ACTIVITY,
    PLAYING_WAIT,
    SKIP_AFTER,
    AutoSync,
    AutoSyncOps,
    DemoFolderUnusable,
    backoff,
)
from src.services.steam_jobs import FetcherUnavailable, JobBusy

CFG = {
    "enabled": True,
    "interval_minutes": 5,
    "idle_check_minutes": 30,
    "pause_when_playing": True,
}


class Env:
    """Everything the loop asks the app for, under the test's control."""

    def __init__(self):
        self.playing = {"playing": False, "detail": ""}
        self.presence_calls = 0
        self.pending: list[str | None] = [None]
        self.folder_error: str | None = None
        self.import_ok = True
        self.import_raises: Exception | None = None
        self.outstanding_values: list[int] = [0]
        self.job_code = 0
        self.job_raises: Exception | None = None
        self.job_calls: list[tuple] = []
        self.last_error: str | None = None

    def _is_playing(self):
        self.presence_calls += 1
        return self.playing

    def _next_unimported(self, set_aside):
        if self.folder_error:
            raise DemoFolderUnusable(self.folder_error)
        return self.pending[0] if len(self.pending) == 1 else self.pending.pop(0)

    def _import_one(self, filename):
        if self.import_raises:
            raise self.import_raises
        return self.import_ok

    def _outstanding(self):
        if len(self.outstanding_values) == 1:
            return self.outstanding_values[0]
        return self.outstanding_values.pop(0)

    def _run_job(self, job_type, args):
        self.job_calls.append((job_type, args))
        if self.job_raises:
            raise self.job_raises
        return self.job_code

    def ops(self):
        return AutoSyncOps(
            load_config=lambda: CFG,
            is_playing=self._is_playing,
            next_unimported=self._next_unimported,
            import_one=self._import_one,
            outstanding=self._outstanding,
            run_job=self._run_job,
            last_job_error=lambda: self.last_error,
        )


@pytest.fixture
def env():
    return Env()


@pytest.fixture
def sync(env):
    return AutoSync(env.ops())


# ---------------------------------------------------------------------------
# Deferring to the player
# ---------------------------------------------------------------------------


def test_someone_playing_pauses_the_loop(sync, env):
    """Downloading 280 MB mid-match costs them ping, and the fetcher's own
    Game Coordinator login can drop them out of the game."""
    env.playing = {"playing": True, "detail": "Main is in CS2"}
    assert sync.step(CFG) == PLAYING_WAIT
    assert sync.snapshot()["phase"] == "paused"


def test_the_presence_check_is_skipped_when_disabled(sync, env):
    env.playing = {"playing": True, "detail": ""}
    sync.step({**CFG, "pause_when_playing": False})
    assert env.presence_calls == 0


def test_an_unknown_presence_answer_does_not_pause(sync, env):
    """Privacy settings can make this unknowable; half a signal still beats none."""
    env.playing = {"playing": None, "detail": "could not determine"}
    assert sync.step(CFG) != PLAYING_WAIT


# ---------------------------------------------------------------------------
# Importing takes priority over downloading
# ---------------------------------------------------------------------------


def test_a_demo_on_disk_is_imported_before_anything_is_downloaded(sync, env):
    """Cheapest useful work first, and it clears the way before another 280 MB
    lands. It also validates the demo folder before fetching into it."""
    env.pending = ["pr1maly_1.dem"]
    env.outstanding_values = [5]

    wait = sync.step(CFG)

    assert env.job_calls == []
    assert wait == CFG["interval_minutes"] * 60
    assert sync.snapshot()["totals"]["imported"] == 1


def test_an_unusable_demo_folder_stops_the_download_loop(sync, env):
    """Otherwise it fetches match after match it can never import."""
    env.folder_error = "No sync folder configured"

    assert sync.step(CFG) == BACKOFF_MAX
    assert sync.snapshot()["phase"] == "error"
    assert "sync folder" in sync.snapshot()["last_error"]


def test_a_failed_import_backs_off_instead_of_retrying_immediately(sync, env):
    env.pending = ["bad.dem"]
    env.import_ok = False

    wait = sync.step(CFG)

    assert wait == backoff(1)
    assert sync.snapshot()["totals"]["failed"] == 1
    assert sync.snapshot()["failure_streak"] == 1


def test_a_demo_that_will_not_parse_is_set_aside(sync, env):
    """Retrying forever would block every other match behind one corrupt file."""
    env.pending = ["corrupt.dem"]
    env.import_ok = False

    for _ in range(SKIP_AFTER):
        wait = sync.step(CFG)

    assert sync.snapshot()["skipped"]["corrupt.dem"] == SKIP_AFTER
    assert wait == 5
    assert any("Set aside" in a["text"] for a in sync.snapshot()["activity"])
    assert "corrupt.dem" in sync.set_aside


def test_a_successful_import_clears_the_skiplist_entry(sync, env):
    env.pending = ["flaky.dem"]
    env.import_ok = False
    sync.step(CFG)
    assert sync.snapshot()["skipped"]["flaky.dem"] == 1

    env.import_ok = True
    sync.step(CFG)

    assert "flaky.dem" not in sync.snapshot()["skipped"]
    assert sync.snapshot()["failure_streak"] == 0


def test_a_user_started_job_makes_the_loop_stand_aside(sync, env):
    """JobBusy means the user pressed a button; auto-sync always yields."""
    env.pending = ["queued.dem"]
    env.import_raises = JobBusy("download")

    assert sync.step(CFG) == BUSY_WAIT
    assert sync.snapshot()["phase"] == "blocked"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def test_one_match_is_downloaded_then_imported_with_no_gap(sync, env):
    """The configured interval separates matches, not the two halves of one."""
    env.outstanding_values = [3]
    # Nothing to import before the download, something after it.
    env.pending = [None, "pr1maly_new.dem"]

    wait = sync.step(CFG)

    assert env.job_calls == [("download", ["fetch.js", "--limit", "1"])]
    assert wait == 0
    assert sync.snapshot()["totals"]["downloaded"] == 1


def test_a_match_that_yields_no_demo_moves_on_without_a_long_backoff(sync, env):
    """Exit 0 with nothing new means expired or already held — ledger progress,
    but no file. Not an error."""
    env.outstanding_values = [1]

    wait = sync.step(CFG)

    assert wait == min(CFG["interval_minutes"] * 60, 60)
    assert sync.snapshot()["totals"]["downloaded"] == 0


def test_a_failed_download_backs_off_and_records_why(sync, env):
    env.outstanding_values = [2]
    env.job_code = 1
    env.last_error = "ECONNRESET talking to the CDN"

    wait = sync.step(CFG)

    assert wait == backoff(1)
    assert sync.snapshot()["phase"] == "error"
    assert "ECONNRESET" in sync.snapshot()["last_error"]


def test_a_missing_fetcher_during_download_is_an_error_not_a_crash(sync, env):
    env.outstanding_values = [1]
    env.job_raises = FetcherUnavailable("Node.js was not found")

    assert sync.step(CFG) == BACKOFF_MAX
    assert sync.snapshot()["phase"] == "error"


def test_stopping_mid_download_returns_immediately(sync, env):
    env.outstanding_values = [1]

    def stop_during(job_type, args):
        sync.request_stop()
        return 0

    env._run_job = stop_during
    sync._ops = env.ops()
    assert sync.step(CFG) == 0


# ---------------------------------------------------------------------------
# Checking for new matches
# ---------------------------------------------------------------------------


def test_nothing_outstanding_asks_steam_for_new_matches(sync, env):
    wait = sync.step(CFG)

    assert env.job_calls == [("check", ["sharecodes.js", "--walk"])]
    assert wait == CFG["idle_check_minutes"] * 60
    assert sync.snapshot()["phase"] == "waiting"


def test_finding_new_matches_starts_on_them_immediately(sync, env):
    env.outstanding_values = [0, 4]

    assert sync.step(CFG) == 0
    assert any("4 match" in a["text"] for a in sync.snapshot()["activity"])


def test_a_failed_check_backs_off(sync, env):
    env.job_code = 2
    env.last_error = "Steam Web API returned 503"

    assert sync.step(CFG) == backoff(1)
    assert "503" in sync.snapshot()["last_error"]


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def test_backoff_doubles_and_is_capped():
    assert backoff(1) == BACKOFF_BASE
    assert backoff(2) == BACKOFF_BASE * 2
    assert backoff(3) == BACKOFF_BASE * 4
    assert backoff(99) == BACKOFF_MAX


def test_backoff_never_returns_zero_or_negative():
    """It must never turn into a hot loop against a failing Steam."""
    for streak in (-5, 0, 1):
        assert backoff(streak) >= BACKOFF_BASE


def test_repeated_failures_never_disable_auto_sync(sync, env):
    """A Steam outage should not silently end a job the user expects to still
    be running when they come back."""
    env.outstanding_values = [1]
    env.job_code = 1

    waits = [sync.step(CFG) for _ in range(8)]

    assert all(w > 0 for w in waits)
    assert waits[-1] == BACKOFF_MAX
    assert sync.snapshot()["phase"] == "error"


# ---------------------------------------------------------------------------
# The activity feed
# ---------------------------------------------------------------------------


def test_activity_feed_is_capped(sync):
    for n in range(MAX_ACTIVITY + 20):
        sync.note(f"note {n}")

    activity = sync.snapshot()["activity"]
    assert len(activity) == MAX_ACTIVITY
    assert activity[-1]["text"] == f"note {MAX_ACTIVITY + 19}"


def test_snapshot_copies_its_collections(sync):
    """The status endpoint serialises this while the loop keeps running."""
    sync.note("one")
    snapshot = sync.snapshot()
    snapshot["activity"].append("injected")
    snapshot["totals"]["imported"] = 999

    assert len(sync.snapshot()["activity"]) == 1
    assert sync.snapshot()["totals"]["imported"] == 0
