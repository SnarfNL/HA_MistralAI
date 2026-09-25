"""Print the Python requirements of the HA components this integration uses.

pytest-homeassistant-custom-component installs Home Assistant itself, but not
the packages its built-in components need (hassil for conversation, mutagen
for tts, ...). This reads them from the installed HA's own manifests, so each
test matrix leg gets the versions that match its HA version:

    python scripts/component_requirements.py > requirements_components.txt
    pip install -r requirements_components.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import homeassistant.components

ROOTS = [
    "ai_task",
    "assist_pipeline",
    "button",
    "conversation",
    "diagnostics",
    "repairs",
    "stt",
    "tts",
]


def main() -> None:
    base = Path(homeassistant.components.__file__).parent
    seen: set[str] = set()
    requirements: set[str] = set()
    todo = list(ROOTS)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        manifest = json.loads((base / name / "manifest.json").read_text(encoding="utf-8"))
        requirements.update(manifest.get("requirements", []))
        todo += manifest.get("dependencies", []) + manifest.get("after_dependencies", [])
    print("\n".join(sorted(requirements)))


if __name__ == "__main__":
    main()
