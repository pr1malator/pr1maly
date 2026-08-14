"""One JSON file under data/, read and written the same way every time.

api.py carried eight near-identical pairs of these — read the file, swallow the
exception, return a default; make the parent directory, dump with indent=2,
append a newline. They had already drifted: some caught JSONDecodeError only,
some caught everything, and a generic `_read_json_file` helper existed but was
defined thirteen hundred lines after the first four copies and used by half of
them.

Two things are deliberately not the old behaviour:

Reads use utf-8-sig. A byte-order mark makes json.loads raise, and these are
files a user opens in an editor — Notepad adds one.

Writes go through a temporary file and os.replace. These hold the Steam match
ledger, the Web API key and LLM keys; losing one to a crash halfway through a
write costs the user credentials they have to go and reissue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.config.settings import DATA_DIR


class JsonStore:
    """A named JSON file in the data directory.

    Args:
        name: filename, e.g. ``"accounts.json"``.
        defaults: merged under whatever is on disk on every read, so a config
            gains new keys without the user having to edit their file.
        private: restrict the file to its owner. Set for anything holding a
            credential — it is a no-op on filesystems without POSIX modes.
    """

    def __init__(
        self,
        name: str,
        *,
        defaults: dict[str, Any] | None = None,
        private: bool = False,
    ) -> None:
        self.name = name
        self.defaults = defaults or {}
        self.private = private

    @property
    def path(self) -> Path:
        # Resolved per call rather than in __init__ so a test that repoints
        # DATA_DIR is honoured by stores that already exist.
        return DATA_DIR / self.name

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> dict[str, Any]:
        """Defaults, overlaid with the file. Missing or unreadable reads as the defaults."""
        data = dict(self.defaults)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return data
        if isinstance(loaded, dict):
            data.update(loaded)
        return data

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2) + "\n"

        tmp = self.path.with_name(f".{self.path.name}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            if self.private:
                # Tighten before the file is reachable under its real name,
                # not after — otherwise there is a window where it is not.
                self._restrict(tmp)
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)

    # -- list-shaped files -------------------------------------------------
    #
    # accounts.json and friends.json wrap their list in a single key rather
    # than being a bare array.

    def read_list(self, key: str) -> list[Any]:
        value = self.read().get(key)
        return value if isinstance(value, list) else []

    def write_list(self, key: str, items: list[Any]) -> None:
        self.write({key: items})

    @staticmethod
    def _restrict(path: Path) -> None:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # filesystem does not support it; nothing better to do
