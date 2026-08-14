"""Where things live on disk.

Every module used to work this out for itself with its own
``Path(__file__).parent / "data" / ...``, which is correct only as long as the
file computing it never moves. Naming it once means moving a module cannot
silently point it at a different directory.

Environment variables (``DB_PATH``, ``DEMO_DIR``, ``STEAM_API_KEY``) are
deliberately still read where they are used. ``DB_PATH`` in particular is read
at import time by src/database.py, and the test suite depends on setting it
before that import — moving it here would change when it is evaluated.
"""

from __future__ import annotations

from pathlib import Path

# src/config/settings.py -> src/config -> src -> the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Everything the user owns: the database, their accounts, Steam tokens, the
# match ledger, AI keys. docker-compose bind-mounts this from the host.
DATA_DIR = PROJECT_ROOT / "data"

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FETCHER_DIR = PROJECT_ROOT / "fetcher"
