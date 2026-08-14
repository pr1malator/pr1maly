"""Tests for the Steam fetcher job supervisor.

This code had no tests at all: module-global mutable state, a lock, a daemon
thread and a subprocess whose stdout is parsed line by line. It is also the part
of the app that spends the user's bandwidth and touches their Steam credentials,
so "it seemed to work" is a poor guarantee.

Written against the module globals before SteamJobRunner existed, then pointed
at the class — the assertions did not change, which is the evidence the move
preserved behaviour.

Nothing here starts a real Node process; subprocess.Popen is replaced with a
fake that plays back canned stdout.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.services import steam_jobs
from src.services.steam_jobs import (
    FetcherUnavailable,
    JobBusy,
    SteamJobRunner,
)

os.environ.setdefault("DB_PATH", ":memory:")

import api  # noqa: E402
from src.api import deps  # noqa: E402

client = TestClient(api.app)


@pytest.fixture
def runner(tmp_path):
    """A fresh slot per test, with the fetcher directory looking installed."""
    (tmp_path / "node_modules").mkdir()
    return SteamJobRunner(tmp_path)


class FakeProc:
    """Stands in for a Node child process."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter(f"{line}\n" for line in lines)
        self.returncode = returncode
        self.waited = False
        self.terminated = False

    def wait(self):
        self.waited = True
        return self.returncode

    def terminate(self):
        self.terminated = True


@pytest.fixture
def spawn(monkeypatch):
    """Script the next subprocess the runner starts."""
    monkeypatch.setattr(SteamJobRunner, "node_path", staticmethod(lambda: "/usr/bin/node"))

    def _spawn(lines, returncode=0):
        proc = FakeProc(lines, returncode)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
        return proc

    return _spawn


# ---------------------------------------------------------------------------
# The single job slot
# ---------------------------------------------------------------------------


def test_claiming_the_slot_marks_it_running(runner):
    runner.claim("check")
    job = runner.snapshot()
    assert job["running"] is True
    assert job["type"] == "check"
    assert job["auto"] is False
    assert job["started_at"]


def test_a_second_claim_is_refused(runner):
    """Two fetchers writing into the same demo folder is the thing to avoid."""
    runner.claim("download")
    with pytest.raises(JobBusy) as excinfo:
        runner.claim("check")
    assert excinfo.value.job_type == "download"
    assert "download" in str(excinfo.value)


def test_a_busy_slot_is_a_409_over_http(monkeypatch):
    """The service raises JobBusy; the HTTP layer owes the caller a 409."""
    monkeypatch.setattr(deps._steam_jobs, "require_available", lambda: None)
    deps._steam_jobs.claim("download")
    try:
        with pytest.raises(HTTPException) as excinfo:
            deps._job_claim("check")
        assert excinfo.value.status_code == 409
    finally:
        deps._steam_jobs.release(0)


def test_auto_jobs_are_flagged_so_the_ui_can_tell_them_apart(runner):
    runner.claim("download", auto=True)
    assert runner.snapshot()["auto"] is True


def test_claiming_clears_the_previous_job_output(runner):
    runner.claim("check")
    runner.log("stale line")
    runner.release(0)

    runner.claim("download")
    job = runner.snapshot()
    assert job["lines"] == []
    assert job["events"] == []
    assert job["exit_code"] is None
    assert job["finished_at"] is None


def test_releasing_records_the_exit_code_and_frees_the_slot(runner):
    runner.claim("check")
    runner.release(3)
    job = runner.snapshot()
    assert job["running"] is False
    assert job["exit_code"] == 3
    assert job["finished_at"]
    runner.claim("check")  # slot is free again


# ---------------------------------------------------------------------------
# The visible log
# ---------------------------------------------------------------------------


def test_log_is_capped_and_keeps_the_newest(runner):
    for n in range(steam_jobs.MAX_JOB_LINES + 50):
        runner.log(f"line {n}")

    lines = runner.snapshot()["lines"]
    assert len(lines) == steam_jobs.MAX_JOB_LINES
    assert lines[-1] == f"line {steam_jobs.MAX_JOB_LINES + 49}"
    assert lines[0] == "line 50"


def test_snapshot_copies_the_lists(runner):
    """The UI polls this; handing out the live lists would let a reader see the
    log mutate underneath it, and let a caller mutate the job record."""
    runner.claim("check")
    runner.log("one")

    snapshot = runner.snapshot()
    snapshot["lines"].append("injected")
    runner.log("two")

    assert "injected" not in runner.snapshot()["lines"]
    assert snapshot["lines"] == ["one", "injected"]


def test_last_error_is_the_final_meaningful_line(runner):
    runner.log("Connecting")
    runner.log("   ")
    runner.log("ECONNRESET talking to the CDN")
    runner.log("")
    assert runner.last_error() == "ECONNRESET talking to the CDN"


def test_last_error_of_a_silent_job_is_none(runner):
    assert runner.last_error() is None


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_missing_node_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setattr(SteamJobRunner, "node_path", staticmethod(lambda: None))
    with pytest.raises(FetcherUnavailable, match="Node"):
        SteamJobRunner(tmp_path).require_available()


def test_missing_node_modules_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(SteamJobRunner, "node_path", staticmethod(lambda: "/usr/bin/node"))
    with pytest.raises(FetcherUnavailable, match="npm install"):
        SteamJobRunner(tmp_path).require_available()  # no node_modules in tmp_path


def test_an_unavailable_fetcher_is_a_400_over_http(monkeypatch):
    def unavailable():
        raise FetcherUnavailable("Node.js was not found on this machine.")

    monkeypatch.setattr(deps._steam_jobs, "require_available", unavailable)
    with pytest.raises(HTTPException) as excinfo:
        deps._require_fetcher()
    assert excinfo.value.status_code == 400


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------


def test_stdout_becomes_the_visible_log(runner, spawn):
    spawn(["Checking account Main", "Found 3 new matches"])
    runner.claim("check")

    assert runner.run(["sharecodes.js", "--walk"]) == 0
    job = runner.snapshot()
    assert job["lines"] == ["Checking account Main", "Found 3 new matches"]
    assert job["running"] is False
    assert job["exit_code"] == 0


def test_structured_events_are_routed_away_from_the_log(runner, spawn):
    """STEAM_EVENT lines drive the QR sign-in handshake and must not be shown
    to the user as log noise."""
    spawn([
        "Starting sign-in",
        'STEAM_EVENT {"type": "qr", "url": "https://s.team/q/1"}',
        "Waiting for confirmation",
    ])
    runner.claim("auth")
    runner.run(["auth-qr.js", "Main"])

    job = runner.snapshot()
    assert job["lines"] == ["Starting sign-in", "Waiting for confirmation"]
    assert job["events"] == [{"type": "qr", "url": "https://s.team/q/1"}]


def test_a_malformed_event_line_is_shown_rather_than_swallowed(runner, spawn):
    """Better in the log than silently dropped — it is the only trace of it."""
    spawn(["STEAM_EVENT {not json"])
    runner.claim("auth")
    runner.run(["auth-qr.js"])

    job = runner.snapshot()
    assert job["events"] == []
    assert job["lines"] == ["STEAM_EVENT {not json"]


def test_a_failing_job_records_its_exit_code(runner, spawn):
    spawn(["Download failed"], returncode=1)
    runner.claim("download")

    assert runner.run(["fetch.js"]) == 1
    assert runner.snapshot()["exit_code"] == 1
    assert runner.snapshot()["running"] is False


def test_the_slot_is_freed_even_when_spawning_blows_up(runner, monkeypatch):
    """Otherwise one failure wedges the slot and every later job is refused."""
    monkeypatch.setattr(SteamJobRunner, "node_path", staticmethod(lambda: "/usr/bin/node"))

    def explode(*args, **kwargs):
        raise OSError("node vanished")

    monkeypatch.setattr(subprocess, "Popen", explode)
    runner.claim("check")

    assert runner.run(["sharecodes.js"]) == -1
    job = runner.snapshot()
    assert job["running"] is False
    assert job["exit_code"] == -1
    assert any("node vanished" in line for line in job["lines"])
    runner.claim("check")  # provably not wedged


def test_the_child_is_waited_on(runner, spawn):
    """Not waiting leaves a zombie behind on POSIX."""
    proc = spawn(["done"])
    runner.claim("check")
    runner.run(["sharecodes.js"])
    assert proc.waited is True


# ---------------------------------------------------------------------------
# Cancelling
# ---------------------------------------------------------------------------


def test_cancelling_when_nothing_runs_is_not_an_error(runner):
    outcome = runner.cancel()
    assert outcome["cancelled"] is False
    assert "No job" in outcome["detail"]


def test_cancelling_terminates_the_child_and_marks_the_job(runner):
    proc = FakeProc([])
    runner.claim("auth")
    runner._proc = proc

    assert runner.cancel()["cancelled"] is True
    assert proc.terminated is True
    assert runner.snapshot()["cancelled"] is True


def test_auto_only_cancel_leaves_a_user_started_job_alone(runner):
    """Switching Auto-Sync off must not kill a download the user began."""
    proc = FakeProc([])
    runner.claim("download", auto=False)
    runner._proc = proc

    assert runner.cancel(only_auto=True)["cancelled"] is False
    assert proc.terminated is False
    assert runner.snapshot()["cancelled"] is False


def test_auto_only_cancel_stops_an_auto_job(runner):
    proc = FakeProc([])
    runner.claim("download", auto=True)
    runner._proc = proc

    assert runner.cancel(only_auto=True)["cancelled"] is True
    assert proc.terminated is True


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


def test_job_endpoint_reports_the_current_job():
    deps._steam_jobs.claim("download")
    deps._steam_jobs.log("Downloading 1 of 3")
    try:
        body = client.get("/api/steam/job").json()
        assert body["running"] is True
        assert body["type"] == "download"
        assert body["lines"] == ["Downloading 1 of 3"]
    finally:
        deps._steam_jobs.release(0)


def test_cancel_endpoint_when_nothing_runs():
    body = client.post("/api/steam/job/cancel").json()
    assert body["cancelled"] is False
    assert "No job" in body["detail"]
