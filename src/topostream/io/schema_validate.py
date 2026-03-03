"""
src/topostream/io/schema_validate.py
======================================
JSON schema validation helper for topology event stream tokens.

Loads the canonical schema once and exposes a simple validate_token()
function.  Raises jsonschema.ValidationError on failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "schemas"
    / "topology_event_stream.schema.json"
)

_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with _SCHEMA_PATH.open() as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def validate_token(token: dict) -> None:
    """Validate a single token against the topology event stream schema.

    Raises ``jsonschema.ValidationError`` if the token is invalid.
    """
    schema = _load_schema()
    jsonschema.validate(token, schema)


def validate_tokens(tokens: list[dict]) -> list[str]:
    """Validate a list of tokens.  Returns list of error messages (empty = all valid)."""
    errors: list[str] = []
    schema = _load_schema()
    for i, tok in enumerate(tokens):
        try:
            jsonschema.validate(tok, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Token {i}: {e.message}")
    return errors
