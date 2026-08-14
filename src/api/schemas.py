"""Request and response models.

Fourteen of these were scattered through api.py — four near the top and the
rest beside whichever route happened to use them. They are the typed part of
the public API, so they are worth being able to read in one place.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class NotesUpdate(BaseModel):
    notes: str

class TagCreate(BaseModel):
    tag: str

class ConfigResponse(BaseModel):
    steam_id: str | None

class ConfigUpdate(BaseModel):
    steam_id: str

class ChatMessageIn(BaseModel):
    message: str
    provider: str | None = None
    model: str | None = None

class AIConfigUpdate(BaseModel):
    providers: dict[str, dict] | None = None
    active_provider: str | None = None
    active_model: str | None = None
    system_instructions: str | None = None
    prompts: list[dict] | None = None

class AccountCreate(BaseModel):
    name: str
    steam_id: str
    display_name: str = ""
    rank: str = ""

class AccountUpdate(BaseModel):
    name: str | None = None
    display_name: str | None = None
    rank: str | None = None

class FriendCreate(BaseModel):
    steam_id: str
    name: str = ""

class SteamApiKeyIn(BaseModel):
    api_key: str
    account: str | None = None

class SteamCodesIn(BaseModel):
    auth_code: str
    share_code: str

class SteamTogglesIn(BaseModel):
    walk_enabled: bool | None = None
    download_enabled: bool | None = None

class SteamDownloadIn(BaseModel):
    limit: int | None = None

class AutoSyncIn(BaseModel):
    enabled: bool | None = None
    interval_minutes: int | None = None
    idle_check_minutes: int | None = None
    pause_when_playing: bool | None = None


class UpdateConfig(BaseModel):
    """Whether the app may ask the project's site for the current version."""

    enabled: bool


class UpdateStatusResponse(BaseModel):
    current: str
    latest: str | None = None
    update_available: bool = False
    notes: str | None = None
    released: str | None = None
    checked_at: str | None = None
    enabled: bool = False
    error: str | None = None
