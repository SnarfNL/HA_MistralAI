"""Web-search conversations against the Mistral Conversations API.

Pure helpers (no Home Assistant imports) so they can be unit-tested directly:

* ``WebSearchConversations`` — bounded HA-conversation → Mistral-conversation map.
* ``build_conversation_payload`` — request body for a conversation that carries
  its own model, tool and instructions, so no Mistral *agent* is needed.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

# Matches HA's own idle timeout for a chat log.
DEFAULT_TTL_SECONDS = 300
DEFAULT_MAX_ITEMS = 50

# Generic on purpose: no entity list, date or user prompt ever ends up here.
WEB_SEARCH_INSTRUCTIONS = (
    "Answer the question factually and concisely in one or two short sentences "
    "suited to being read aloud. Do not use lists, markdown or citations."
)


class WebSearchConversations:
    """Maps an HA ``conversation_id`` to a Mistral conversation id.

    Entries expire ``ttl`` seconds after their last use and the map never holds
    more than ``max_items`` entries (oldest evicted first). Methods return the
    Mistral ids that left the map, so the caller can delete them server-side.
    """

    def __init__(
        self,
        ttl: float = DEFAULT_TTL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._max_items = max_items
        self._clock = clock
        # ha_conversation_id -> (mistral_conversation_id, last_used)
        self._items: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def pop_expired(self) -> list[str]:
        """Remove and return the Mistral ids of entries past their TTL."""
        now = self._clock()
        expired = [
            key for key, (_, used) in self._items.items() if now - used >= self._ttl
        ]
        return [self._items.pop(key)[0] for key in expired]

    def get(self, ha_id: str) -> str | None:
        """Return the Mistral id for ``ha_id``, or ``None`` if unknown or expired."""
        item = self._items.get(ha_id)
        if item is None:
            return None
        mistral_id, used = item
        if self._clock() - used >= self._ttl:
            return None
        return mistral_id

    def set(self, ha_id: str, mistral_id: str) -> list[str]:
        """Store or refresh an entry; return ids evicted by the size cap."""
        replaced = self._items.pop(ha_id, None)
        self._items[ha_id] = (mistral_id, self._clock())
        evicted: list[str] = []
        if replaced and replaced[0] != mistral_id:
            evicted.append(replaced[0])
        while len(self._items) > self._max_items:
            _, (old_id, _) = self._items.popitem(last=False)
            evicted.append(old_id)
        return evicted

    def clear(self) -> list[str]:
        """Remove everything; return all Mistral ids that were held."""
        ids = [mistral_id for mistral_id, _ in self._items.values()]
        self._items.clear()
        return ids


def build_conversation_payload(
    model: str,
    user_text: str,
    language: str | None,
    *,
    store: bool,
) -> dict[str, Any]:
    """Body for ``POST /v1/conversations`` with web search and no agent."""
    instructions = WEB_SEARCH_INSTRUCTIONS
    if language:
        instructions += f" Reply in the language with code '{language}'."
    return {
        "model": model,
        "instructions": instructions,
        "tools": [{"type": "web_search"}],
        "inputs": user_text,
        "store": store,
    }
