"""Configuration: where files live, and how the JSON ones are read and written."""

from src.config.settings import DATA_DIR, PROJECT_ROOT
from src.config.store import JsonStore

__all__ = ["DATA_DIR", "PROJECT_ROOT", "JsonStore"]
