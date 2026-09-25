"""Which Mistral models exist, and what to suggest when one is retired (MA-14)."""
from __future__ import annotations

from typing import Any


def model_available(model: str, models: list[dict[str, Any]]) -> bool:
    """True when *model* is an id or alias in a GET /v1/models list."""
    for item in models:
        aliases = item.get("aliases")
        if model == item.get("id") or (isinstance(aliases, list) and model in aliases):
            return True
    return False
