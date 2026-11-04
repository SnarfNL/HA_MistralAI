"""Probe how Mistral's web search behaves on /v1/conversations (issue #88).

Usage (never in CI):  MISTRAL_API_KEY=... python scripts/probe_web_search.py
Optional: PROBE_MODEL=mistral-medium-latest (the default) to try another model.

COST: web search is billed per search. This script makes 9 API calls (6
conversations incl. one on a temporary agent, 1 agent create, 2 deletes), and
each conversation that really searches is billed as a web search. Run it once,
not in a loop.
The API key is read from the environment and never printed or written.

What it does: asks one current-events question that a model cannot answer
from memory, in several variants, and prints one table row per variant:

* inline conversation (model + tools, what the integration does today), with
  ``store`` true and false;
* inline, with ``completion_args.tool_choice`` ``auto`` and ``required``;
* inline, with instructions that say to always use the web search tool;
* agent-based: a temporary agent with the web_search tool, then a conversation
  that uses its ``agent_id``.

Every conversation and agent it creates is deleted again, also on failure.
The columns tell which variant really runs a search: ``tool.execution`` says
whether the answer contains a tool execution, ``billed`` is
``usage.connectors.web_search`` when Mistral reports it.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

BASE = "https://api.mistral.ai/v1"
KEY = os.environ.get("MISTRAL_API_KEY", "")
MODEL = os.environ.get("PROBE_MODEL", "mistral-medium-latest")
QUESTION = "What is today's top headline on bbc.com/news? Name the headline."

# Same text as the integration's WEB_SEARCH_INSTRUCTIONS.
GENERIC_INSTRUCTIONS = (
    "Answer the question factually and concisely in one or two short sentences "
    "suited to being read aloud. Do not use lists, markdown or citations."
)
STRICT_INSTRUCTIONS = (
    "You have no knowledge of current events. Always use the web search tool "
    "before you answer, even if you think you know the answer. "
    + GENERIC_INSTRUCTIONS
)
WEB_SEARCH_TOOL = {"type": "web_search"}


def call(method: str, path: str, body: dict | None = None) -> tuple[int | str, Any]:
    """Send one request; return (HTTP status or 'ERR', parsed JSON or text)."""
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw, status = resp.read().decode("utf-8", "replace"), resp.status
    except urllib.error.HTTPError as err:
        raw, status = err.read().decode("utf-8", "replace"), err.code
    except (urllib.error.URLError, TimeoutError) as err:
        return "ERR", str(getattr(err, "reason", err))
    try:
        return status, json.loads(raw) if raw else {}
    except ValueError:
        return status, raw


def answer_text(data: Any) -> str:
    """The assistant's text from a conversation response, one line."""
    if not isinstance(data, dict):
        return str(data)
    outputs = data.get("outputs")
    parts: list[str] = []
    for output in outputs if isinstance(outputs, list) else []:
        if not isinstance(output, dict) or output.get("type") == "tool.execution":
            continue
        content = output.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
    text = "".join(parts) or str(data.get("message", ""))
    return " ".join(text.split())


def ran_search(data: Any) -> bool:
    outputs = data.get("outputs") if isinstance(data, dict) else None
    return isinstance(outputs, list) and any(
        isinstance(o, dict) and o.get("type") == "tool.execution" for o in outputs
    )


def billed(data: Any) -> str:
    usage = data.get("usage") if isinstance(data, dict) else None
    connectors = usage.get("connectors") if isinstance(usage, dict) else None
    if isinstance(connectors, dict) and "web_search" in connectors:
        return str(connectors["web_search"])
    return "-"


def inline_body(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "instructions": GENERIC_INSTRUCTIONS,
        "tools": [WEB_SEARCH_TOOL],
        "inputs": QUESTION,
        "store": False,
    }
    body.update(changes)
    return body


def main() -> int:
    if not KEY:
        print("Set MISTRAL_API_KEY first.")
        return 1

    cleanups: list[Callable[[], tuple[int | str, Any]]] = []
    failed = 0
    rows: list[tuple[str, str, str, str, str]] = []

    def record(variant: str, status: int | str, data: Any) -> None:
        text = answer_text(data)[:120]
        rows.append(
            (
                variant,
                str(status),
                "yes" if ran_search(data) else "no",
                billed(data),
                text,
            )
        )

    def delete_later(kind: str, ident: Any) -> None:
        if isinstance(ident, str):
            cleanups.append(lambda: call("DELETE", f"/{kind}/{ident}"))

    def conversation(variant: str, body: dict[str, Any]) -> None:
        status, data = call("POST", "/conversations", body)
        record(variant, status, data)
        # Only a stored conversation exists on Mistral's side afterwards.
        if body.get("store") and isinstance(data, dict):
            delete_later("conversations", data.get("conversation_id"))

    try:
        conversation("inline, store=true", inline_body(store=True))
        conversation("inline, store=false", inline_body())
        for choice in ("auto", "required"):
            conversation(
                f"inline, tool_choice={choice}",
                inline_body(completion_args={"tool_choice": choice}),
            )
        conversation(
            "inline, strict instructions",
            inline_body(instructions=STRICT_INSTRUCTIONS),
        )

        status, agent = call(
            "POST",
            "/agents",
            {
                "model": MODEL,
                "name": "probe-web-search (temporary)",
                "instructions": GENERIC_INSTRUCTIONS,
                "tools": [WEB_SEARCH_TOOL],
            },
        )
        agent_id = agent.get("id") if isinstance(agent, dict) else None
        if isinstance(agent_id, str):
            delete_later("agents", agent_id)
            conversation(
                "agent-based",
                {"agent_id": agent_id, "inputs": QUESTION, "store": False},
            )
        else:
            record("agent-based (agent not created)", status, agent)
    finally:
        for cleanup in reversed(cleanups):
            status, _ = cleanup()
            if status not in (200, 204, 404):
                failed += 1

    header = ("variant", "HTTP", "tool.execution", "billed", "answer (first 120 chars)")
    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header) - 1)]
    for row in table:
        print(
            "  ".join(cell.ljust(w) for cell, w in zip(row, widths)) + "  " + row[-1]
        )
    print(f"\nModel: {MODEL}. Question: {QUESTION}")
    if failed:
        print(f"WARNING: {failed} cleanup call(s) failed; check Mistral Studio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
