"""Helpers for unit tests that build entities without setting HA up."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from custom_components.mistral_conversation import MistralRuntimeData
from custom_components.mistral_conversation.api import MistralClient

API_KEY = "sk-test-key"


def attach_runtime(entry: Any, session: Any = None, **fields: Any) -> MistralRuntimeData:
    """Give a stand-in config entry real runtime data around a fake *session*."""
    if not hasattr(entry, "async_start_reauth"):
        entry.async_start_reauth = MagicMock()
    errors: deque[dict[str, Any]] = deque(maxlen=10)
    runtime = MistralRuntimeData(
        client=MistralClient(SimpleNamespace(), entry, session, API_KEY, errors),
        errors=errors,
        **fields,
    )
    entry.runtime_data = runtime
    return runtime
