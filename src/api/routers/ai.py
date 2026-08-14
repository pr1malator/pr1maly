"""AI provider configuration."""

from __future__ import annotations

from fastapi import APIRouter

from src.ai_service import (
    PROVIDERS as AI_PROVIDERS,
    load_config as load_ai_config,
    mask_key,
    save_config as save_ai_config,
)
from src.api.schemas import (
    AIConfigUpdate,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# AI Config
# ---------------------------------------------------------------------------
@router.get("/api/ai/providers")
def list_ai_providers():
    """Return available AI providers with their model lists."""
    return AI_PROVIDERS

@router.get("/api/ai/config")
def get_ai_config():
    """Return current AI configuration (API keys masked)."""
    config = load_ai_config()
    safe = {
        "providers": {},
        "active_provider": config.get("active_provider", ""),
        "active_model": config.get("active_model", ""),
        "system_instructions": config.get("system_instructions", ""),
        "prompts": config.get("prompts", []),
    }
    for name, prov in config.get("providers", {}).items():
        safe["providers"][name] = {
            "api_key_set": bool(prov.get("api_key", "").strip()),
            "api_key_masked": mask_key(prov.get("api_key", "")),
            "default_model": prov.get("default_model", ""),
        }
    return safe

@router.put("/api/ai/config")
def update_ai_config(body: AIConfigUpdate):
    """Update AI configuration (providers, keys, prompts)."""
    config = load_ai_config()
    if body.providers is not None:
        existing = config.get("providers", {})
        for name, prov in body.providers.items():
            key = prov.get("api_key", "").strip()
            if key:
                existing.setdefault(name, {})["api_key"] = key
            if "default_model" in prov:
                existing.setdefault(name, {})["default_model"] = prov["default_model"]
        config["providers"] = existing
    if body.active_provider is not None:
        config["active_provider"] = body.active_provider
    if body.active_model is not None:
        config["active_model"] = body.active_model
    if body.system_instructions is not None:
        config["system_instructions"] = body.system_instructions
    if body.prompts is not None:
        config["prompts"] = body.prompts
    save_ai_config(config)
    return {"status": "ok"}
