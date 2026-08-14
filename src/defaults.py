"""The data/ scaffold that ships with a release.

A downloaded release contains a starter ``data/`` directory so the app has
something sane to read on first launch. Those templates used to be PowerShell
here-strings inside build-release.ps1 — a third copy of values that also live
in api.py and ai_service.py, and one that had already drifted: the shipped
ai_config.json named ``gpt-4o-mini`` while the app's own provider table had
moved on to ``gpt-5.4-mini``, so every new user started out pointed at an older
model than the app recommends.

Deriving them here means that cannot happen again. What stays hardcoded is only
what is genuinely scaffold — the placeholder account name, the container demo
path — and not anything the running app also has an opinion about.

Nothing in the running application reads this module; it exists for
tools/build_release.py. The app's own fallbacks (used when a file is absent)
stay where they are, in api.py and ai_service.py.
"""

from __future__ import annotations

import json
from typing import Any

from src.ai_service import DEFAULT_CONFIG as AI_DEFAULT_CONFIG, PROVIDERS
from src.services.updates import LATEST_URL as UPDATE_URL

# The provider the scaffold pre-selects. Its key is left empty — the point is to
# show the shape, not to guess which vendor someone uses.
_SCAFFOLD_PROVIDER = "openai"

# Where demos live inside the container. docker-compose mounts the host's CS2
# replay folder here, so this is the correct default for a Docker install and a
# visible thing to change for a bare one.
_CONTAINER_DEMO_DIR = "/demos"


def _ai_config() -> dict[str, Any]:
    """Starter AI config: the app's own prompts, no API key, current model."""
    default_model = PROVIDERS[_SCAFFOLD_PROVIDER]["default_model"]
    config = json.loads(json.dumps(AI_DEFAULT_CONFIG))  # deep copy
    config["providers"] = {
        _SCAFFOLD_PROVIDER: {"api_key": "", "default_model": default_model}
    }
    config["active_provider"] = _SCAFFOLD_PROVIDER
    config["active_model"] = default_model
    return config


def release_data_files() -> dict[str, Any]:
    """Filename -> JSON content for the shipped ``data/`` directory.

    Deliberately absent, and it must stay that way: steam_tokens.json (Steam
    refresh tokens are password-equivalent), steam_sharecodes.json (the match
    ledger and the Web API key), onboarding.json (so a fresh install actually
    runs onboarding), and the database itself.
    """
    return {
        # Off, and shipped that way explicitly rather than relying on the
        # in-app default: this is the only thing in the release that would
        # contact a server the user did not configure, so a downloader should
        # be able to see its state without running anything.
        "update_config.json": {
            "enabled": False,
            "url": UPDATE_URL,
        },
        "accounts.json": {
            "accounts": [
                {
                    "name": "MyAccount",
                    "steam_id": "",
                    "display_name": "",
                    "rank": "",
                    "active": True,
                }
            ]
        },
        "ai_config.json": _ai_config(),
        "ai_roles.json": {},
        "friends.json": {"friends": []},
        "sync_config.json": {"folder": _CONTAINER_DEMO_DIR},
    }
