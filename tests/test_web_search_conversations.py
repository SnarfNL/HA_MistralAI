"""Tests for the bounded web-search conversation map and request payload."""
# ruff: noqa: I001 - `_ha_stubs` must be imported before `mistral_conversation`.
from __future__ import annotations

import unittest

from . import _ha_stubs  # noqa: F401  side-effect: install HA stubs

from mistral_conversation._web_search import (
    WEB_SEARCH_INSTRUCTIONS,
    WebSearchConversations,
    build_conversation_payload,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class WebSearchConversationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.convs = WebSearchConversations(ttl=300, max_items=3, clock=self.clock)

    def test_unknown_id_is_none(self) -> None:
        self.assertIsNone(self.convs.get("ha-1"))

    def test_set_then_get(self) -> None:
        self.assertEqual(self.convs.set("ha-1", "m-1"), [])
        self.assertEqual(self.convs.get("ha-1"), "m-1")

    def test_entry_expires_after_ttl(self) -> None:
        self.convs.set("ha-1", "m-1")
        self.clock.now += 299
        self.assertEqual(self.convs.get("ha-1"), "m-1")
        self.clock.now += 1
        self.assertIsNone(self.convs.get("ha-1"))

    def test_pop_expired_returns_and_removes_only_expired(self) -> None:
        self.convs.set("old", "m-old")
        self.clock.now += 200
        self.convs.set("new", "m-new")
        self.clock.now += 150  # old is 350s idle, new 150s
        self.assertEqual(self.convs.pop_expired(), ["m-old"])
        self.assertEqual(len(self.convs), 1)
        self.assertEqual(self.convs.get("new"), "m-new")
        self.assertEqual(self.convs.pop_expired(), [])

    def test_set_refreshes_ttl(self) -> None:
        self.convs.set("ha-1", "m-1")
        self.clock.now += 250
        self.convs.set("ha-1", "m-1")
        self.clock.now += 250
        self.assertEqual(self.convs.get("ha-1"), "m-1")

    def test_cap_evicts_oldest_first(self) -> None:
        for i in range(3):
            self.convs.set(f"ha-{i}", f"m-{i}")
            self.clock.now += 1
        self.assertEqual(self.convs.set("ha-3", "m-3"), ["m-0"])
        self.assertEqual(len(self.convs), 3)
        self.assertIsNone(self.convs.get("ha-0"))
        self.assertEqual(self.convs.get("ha-3"), "m-3")

    def test_replacing_with_new_mistral_id_evicts_the_old_one(self) -> None:
        self.convs.set("ha-1", "m-1")
        self.assertEqual(self.convs.set("ha-1", "m-2"), ["m-1"])
        self.assertEqual(self.convs.get("ha-1"), "m-2")

    def test_clear_returns_all_ids(self) -> None:
        self.convs.set("a", "m-a")
        self.convs.set("b", "m-b")
        self.assertEqual(sorted(self.convs.clear()), ["m-a", "m-b"])
        self.assertEqual(len(self.convs), 0)


class BuildPayloadTests(unittest.TestCase):
    def test_uses_model_and_web_search_tool_without_agent(self) -> None:
        p = build_conversation_payload("mistral-small-latest", "weer?", "nl", store=True)
        self.assertEqual(p["model"], "mistral-small-latest")
        self.assertEqual(p["tools"], [{"type": "web_search"}])
        self.assertEqual(p["inputs"], "weer?")
        self.assertNotIn("agent_id", p)

    def test_store_flag_is_passed_through(self) -> None:
        self.assertTrue(build_conversation_payload("m", "q", None, store=True)["store"])
        self.assertFalse(build_conversation_payload("m", "q", None, store=False)["store"])

    def test_instructions_are_generic_plus_language(self) -> None:
        p = build_conversation_payload("m", "q", "nl", store=False)
        self.assertTrue(p["instructions"].startswith(WEB_SEARCH_INSTRUCTIONS))
        self.assertIn("'nl'", p["instructions"])

    def test_no_language_leaves_instructions_generic(self) -> None:
        p = build_conversation_payload("m", "q", None, store=False)
        self.assertEqual(p["instructions"], WEB_SEARCH_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
