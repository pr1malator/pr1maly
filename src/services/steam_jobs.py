"""The single job slot for the Node fetcher.

One fetcher runs at a time. That is the whole point of this class: two of them
writing into the same demo folder is the failure it exists to prevent, and the
slot is also how Auto-Sync yields to anything the user starts by hand.

This was module-global state in api.py — a dict, a lock, a subprocess handle and
seven functions reaching for them through `global`. As a class the state has an
owner, and a test can build one rather than reset four module attributes.

No FastAPI here. The two failure modes raise :class:`JobBusy` and
:class:`FetcherUnavailable`; the HTTP layer decides they are 409 and 400. That
also reads better at the call site than checking ``exc.status_code == 409``,
which is what the Auto-Sync loop used to do.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Lines beginning with this carry structured data for the UI rather than text
# for the user — currently the QR sign-in handshake.
STEAM_EVENT_PREFIX = "STEAM_EVENT "

MAX_JOB_LINES = 500


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobBusy(Exception):
    """Something already holds the job slot."""

    def __init__(self, job_type: str | None) -> None:
        self.job_type = job_type
        super().__init__(f"A {job_type} job is already running.")


class FetcherUnavailable(Exception):
    """Node, or the fetcher's dependencies, are not installed."""


class SteamJobRunner:
    """Runs one fetcher script at a time, streaming its output into a record."""

    def __init__(self, fetcher_dir: Path, max_lines: int = MAX_JOB_LINES) -> None:
        self._fetcher_dir = Path(fetcher_dir)
        self._max_lines = max_lines
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: dict[str, Any] = self._empty_job()

    @staticmethod
    def _empty_job() -> dict[str, Any]:
        return {
            "running": False,
            "type": None,
            # Marks a job started by the Auto-Sync loop rather than a button, so
            # the UI can tell the two apart.
            "auto": False,
            "lines": [],
            "events": [],
            "exit_code": None,
            "cancelled": False,
            "started_at": None,
            "finished_at": None,
        }

    # -- availability ------------------------------------------------------

    @staticmethod
    def node_path() -> str | None:
        return shutil.which("node")

    def require_available(self) -> None:
        """Everything the Node companion needs before a job can start."""
        if not self.node_path():
            raise FetcherUnavailable(
                "Node.js was not found on this machine. The Steam fetcher needs Node 18+."
            )
        if not (self._fetcher_dir / "node_modules").is_dir():
            raise FetcherUnavailable(
                "Fetcher dependencies are not installed. Run: cd fetcher && npm install"
            )

    # -- the slot ----------------------------------------------------------

    def claim(self, job_type: str, auto: bool = False) -> None:
        """Take the slot, or raise :class:`JobBusy`.

        Auto-Sync relies on this being the only way in: a user pressing a button
        while the loop is mid-download gets a clean refusal rather than two
        fetchers writing into the same demo folder.
        """
        with self._lock:
            if self._job["running"]:
                raise JobBusy(self._job["type"])
            self._job = self._empty_job()
            self._job.update(
                running=True, type=job_type, auto=auto, started_at=utc_now()
            )

    def release(self, exit_code: int) -> None:
        with self._lock:
            self._proc = None
            self._job["running"] = False
            self._job["exit_code"] = exit_code
            self._job["finished_at"] = utc_now()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._job["running"])

    # -- output ------------------------------------------------------------

    def log(self, text: str) -> None:
        """Append one line to the visible log, trimming the oldest."""
        with self._lock:
            self._job["lines"].append(text)
            if len(self._job["lines"]) > self._max_lines:
                del self._job["lines"][: -self._max_lines]

    def snapshot(self) -> dict[str, Any]:
        """A copy the UI can poll without seeing the log mutate underneath it."""
        with self._lock:
            return dict(
                self._job,
                lines=list(self._job["lines"]),
                events=list(self._job["events"]),
            )

    def last_error(self) -> str | None:
        """The most useful-looking line from the job that just failed."""
        with self._lock:
            lines = [line.strip() for line in self._job["lines"] if line.strip()]
        return lines[-1][:300] if lines else None

    # -- running -----------------------------------------------------------

    def run(self, args: list[str]) -> int:
        """Run a fetcher script to completion, streaming stdout into the record.

        Assumes the slot is already claimed. Releases it on the way out — including
        when spawning fails, which would otherwise wedge the slot permanently and
        make every later job report busy.
        """
        exit_code = -1
        try:
            proc = subprocess.Popen(
                [self.node_path(), *args],
                cwd=str(self._fetcher_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._proc = proc

            for line in proc.stdout:
                text = line.rstrip("\n")
                if self._consume_event(text):
                    continue
                self.log(text)

            proc.wait()
            exit_code = proc.returncode
        except Exception as exc:
            self.log(f"Failed to run the fetcher: {exc}")
        finally:
            self.release(exit_code)
        return exit_code

    def _consume_event(self, text: str) -> bool:
        """Route a structured line into events. False means "show it instead"."""
        if not text.startswith(STEAM_EVENT_PREFIX):
            return False
        try:
            event = json.loads(text[len(STEAM_EVENT_PREFIX):])
        except ValueError:
            return False
        if event is None:
            return False
        with self._lock:
            self._job["events"].append(event)
        return True

    def start(self, job_type: str, args: list[str]) -> dict[str, Any]:
        """Claim the slot and run the job on a background thread."""
        self.require_available()
        self.claim(job_type)
        threading.Thread(target=self.run, args=(args,), daemon=True).start()
        return self.snapshot()

    def run_sync(self, job_type: str, args: list[str]) -> int:
        """Same job, run to completion on the calling thread. Used by Auto-Sync."""
        self.require_available()
        self.claim(job_type, auto=True)
        return self.run(args)

    def cancel(self, *, only_auto: bool = False) -> dict[str, Any]:
        """Stop the running job.

        Mainly for QR sign-in: dismissing the code should end the attempt straight
        away rather than leaving it pending until Steam's own timeout.

        *only_auto* restricts it to a job the Auto-Sync loop started. Switching
        Auto-Sync off must not kill a download the user began by hand.
        """
        with self._lock:
            running = bool(self._job["running"])
            if only_auto and not self._job.get("auto"):
                running = False
            proc = self._proc if running else None
            if running:
                self._job["cancelled"] = True

        if not running or proc is None:
            return {"cancelled": False, "detail": "No job is running."}

        try:
            proc.terminate()
        except OSError as exc:
            return {"cancelled": False, "detail": f"Could not stop it: {exc}"}
        return {"cancelled": True}
