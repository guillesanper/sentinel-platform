"""JSON schemas for data validation."""

from pathlib import Path

SCHEMA_DIR = Path(__file__).parent

EVENT_SCHEMA_PATH = SCHEMA_DIR / "event.json"


def get_event_schema() -> dict:
    """Load and return the event JSON schema."""
    import json

    with open(EVENT_SCHEMA_PATH) as f:
        return json.load(f)


__all__ = ["get_event_schema", "EVENT_SCHEMA_PATH", "SCHEMA_DIR"]
