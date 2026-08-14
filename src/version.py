"""The application's version, from one place.

It was written twice — pyproject.toml and the FastAPI app — and only the first
was checked against the git tag when a release was cut. Two numbers that can
disagree are worse than one that is occasionally wrong, and they become worse
still once something compares the running version against a published one to
decide whether an update exists.

Read from the installed package metadata where that exists, and from
pyproject.toml where it does not: the Docker image installs requirements.txt
rather than the package itself, but it copies the whole tree, so the file is
there.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version as _installed_version

from src.config.settings import PROJECT_ROOT

_DISTRIBUTION = "pr1maly"


def _read_version() -> str:
    try:
        return _installed_version(_DISTRIBUTION)
    except PackageNotFoundError:
        pass

    pyproject = PROJECT_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        # Neither installed nor shipped with its metadata. Say so rather than
        # inventing a number that an update check would then compare against.
        return "0.0.0+unknown"


APP_VERSION = _read_version()


def parse(version: str) -> tuple[int, ...]:
    """The numeric part of a version, for comparison. Unparseable reads as 0.

    Deliberately not a full PEP 440 or semver implementation: this exists to
    answer "is the published one newer than mine", and releases here are plain
    MAJOR.MINOR.PATCH.
    """
    numbers = []
    for part in version.split("+")[0].split("-")[0].split("."):
        digits = "".join(c for c in part if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers) or (0,)


def is_newer(candidate: str, than: str) -> bool:
    return parse(candidate) > parse(than)
