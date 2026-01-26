"""JSON schemas for data validation."""

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent

EVENT_SCHEMA_PATH = SCHEMA_DIR / "event.json"


def get_event_schema() -> dict:
    """Load and return the event JSON schema."""
    with EVENT_SCHEMA_PATH.open() as f:
        return json.load(f)


__all__ = ["EVENT_SCHEMA_PATH", "SCHEMA_DIR", "get_event_schema"]
