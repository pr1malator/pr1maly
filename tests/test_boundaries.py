"""The layer boundaries ARCHITECTURE.md claims, asserted rather than hoped for.

Each of these was true when it was written and stops being true the first time
someone reaches across a layer because it was quicker. That is not a bug on the
day it happens — it is a bug six months later, when the reason the layers exist
has been forgotten and the doc describing them is fiction.

None of these forbid anything that cannot be done another way; each names the
other way in its failure message.
"""

from __future__ import annotations

import ast
import re

import pytest

from src.config.settings import PROJECT_ROOT

DOMAIN = PROJECT_ROOT / "src" / "domain"
SERVICES = PROJECT_ROOT / "src" / "services"
API = PROJECT_ROOT / "src" / "api"

# SELECT is matched case-sensitively, so English prose about selecting something
# is not a violation; a real query is written in caps like every other one here.
# The rest are two-word phrases that do not occur in prose either way.
_SQL = (
    re.compile(r"\bSELECT\b"),
    re.compile(
        r"\b(?:INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+\w+\s+SET|CREATE\s+TABLE)\b", re.I
    ),
)


def _modules(directory):
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path) -> set[str]:
    """Top-level module names imported anywhere in the file, including locally."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", _modules(SERVICES), ids=lambda p: p.name)
def test_services_do_not_import_the_web_framework(path):
    """A service raises its own exception; the router turns it into a status code.

    src/services/steam_jobs.py used to raise HTTPException(409) from inside the
    job supervisor, so the Auto-Sync loop — which is not serving anyone — had to
    inspect exc.status_code to find out it was busy.
    """
    offenders = sorted(
        name for name in _imported_modules(path)
        if name.split(".")[0] in {"fastapi", "starlette"}
    )
    assert not offenders, (
        f"{path.name} imports {offenders}. Raise a domain exception instead and "
        f"let the router translate it."
    )


@pytest.mark.parametrize("path", _modules(DOMAIN), ids=lambda p: p.name)
def test_the_domain_reaches_nothing_outside_itself(path):
    """Pure logic: given data, returns values. No handle, no request, no process."""
    offenders = sorted(
        name for name in _imported_modules(path)
        if name.split(".")[0] in {
            "fastapi", "starlette", "sqlite3", "requests", "httpx", "subprocess",
        }
        or name.startswith(("src.api", "src.services", "src.database"))
    )
    assert not offenders, (
        f"src/domain/{path.name} imports {offenders}. Metrics take a "
        f"MetricContext; whatever this needs should be read for it and passed in."
    )


@pytest.mark.parametrize("path", _modules(API), ids=lambda p: p.name)
def test_no_sql_in_the_http_layer(path):
    """Queries live in src/database.py, where the batch readers can find them.

    The N+1 that made /api/performance issue 401 queries over 200 matches was a
    loop in the route handler, invisible from the storage layer that could have
    answered it in one.
    """
    hits = [
        f"line {i}: {line.strip()[:70]}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if any(pattern.search(line) for pattern in _SQL)
    ]
    assert not hits, (
        f"src/api/{path.name} contains SQL:\n  " + "\n  ".join(hits) +
        "\nGive it a named function in src/database.py instead."
    )


def test_shared_state_does_not_depend_on_the_routers_at_import_time():
    """deps.py is the composition root's state; routers import it, not the reverse.

    One deliberate exception: the Auto-Sync loop's import step calls the same
    code path as the upload route, and resolves it lazily inside the function.
    """
    tree = ast.parse((API / "deps.py").read_text(encoding="utf-8"))
    top_level = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "src.api.routers"
        )
    ]
    assert not top_level, (
        "src/api/deps.py imports a router at module level, which makes the "
        "import order circular. Import it inside the function that needs it."
    )
