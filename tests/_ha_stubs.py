"""Minimal Home Assistant stubs for unit tests.

Imported for side effects only: registers stubs in ``sys.modules`` covering
every dotted path the integration imports, then prepends ``custom_components``
to ``sys.path`` so the package becomes importable. After this module is
imported, ``import mistral_conversation.<sub>`` works without HA installed.

Strategy:

* Every HA path is a ``MagicMock`` by default — auto-creates attributes,
  works for ``from x import y`` and call-site usage.
* Names that the integration **subclasses** must be real types, otherwise
  the class statement fails with ``TypeError`` (you can't inherit from a
  Mock instance).
* Names checked with ``isinstance`` must also be real types.
* ``ConfigFlow`` is subclassed with a class kwarg (``domain=DOMAIN``); its
  ``__init_subclass__`` must accept ``**kwargs``.
* Exception classes raised or caught by the code must be real ``Exception``
  subclasses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


def _empty_class(name: str) -> type:
    return type(name, (), {})


def _kwarg_class(name: str) -> type:
    """Real type that accepts arbitrary class-statement keyword arguments."""

    class _Stub:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__()

    _Stub.__name__ = name
    return _Stub


def _exception_class(name: str) -> type:
    return type(name, (Exception,), {})


# ---------------------------------------------------------------------------
# Module skeleton
# ---------------------------------------------------------------------------

_PATHS = [
    "aiohttp", "voluptuous", "voluptuous_openapi",
    "homeassistant", "homeassistant.components",
    "homeassistant.components.conversation",
    "homeassistant.components.stt",
    "homeassistant.components.tts",
    "homeassistant.config_entries", "homeassistant.const", "homeassistant.core",
    "homeassistant.data_entry_flow", "homeassistant.exceptions",
    "homeassistant.helpers", "homeassistant.helpers.aiohttp_client",
    "homeassistant.helpers.config_validation",
    "homeassistant.helpers.device_registry",
    "homeassistant.helpers.entity_platform",
    "homeassistant.helpers.intent", "homeassistant.helpers.llm",
    "homeassistant.helpers.selector", "homeassistant.helpers.typing",
]
for _p in _PATHS:
    if _p not in sys.modules:
        sys.modules[_p] = MagicMock(__name__=_p)


# ---------------------------------------------------------------------------
# Real types for subclassed names + isinstance() targets
# ---------------------------------------------------------------------------

_conv = sys.modules["homeassistant.components.conversation"]
for _n in (
    "ConversationEntity", "ConversationEntityFeature",
    "ConversationInput", "ConversationResult", "SystemContent",
    "UserContent", "AssistantContent", "ToolResultContent", "ChatLog",
):
    setattr(_conv, _n, _empty_class(_n))
_conv.ConverseError = _exception_class("ConverseError")

_stt = sys.modules["homeassistant.components.stt"]
for _n in (
    "SpeechToTextEntity", "SpeechMetadata", "SpeechResult",
    "AudioBitRates", "AudioChannels", "AudioCodecs", "AudioFormats",
    "AudioSampleRates", "SpeechResultState",
):
    setattr(_stt, _n, _empty_class(_n))

_tts = sys.modules["homeassistant.components.tts"]
for _n in (
    "TextToSpeechEntity", "TTSAudioRequest", "TTSAudioResponse",
    "TtsAudioType",
):
    setattr(_tts, _n, _empty_class(_n))


class _Voice:
    def __init__(self, voice_id: str = "", name: str = "") -> None:
        self.voice_id = voice_id
        self.name = name


_tts.Voice = _Voice

_cfg = sys.modules["homeassistant.config_entries"]
_cfg.ConfigFlow = _kwarg_class("ConfigFlow")
_cfg.OptionsFlow = _empty_class("OptionsFlow")
_cfg.ConfigEntry = _empty_class("ConfigEntry")

sys.modules["homeassistant.core"].HomeAssistant = _empty_class("HomeAssistant")

_exc = sys.modules["homeassistant.exceptions"]
_exc.HomeAssistantError = _exception_class("HomeAssistantError")
_exc.ConfigEntryAuthFailed = _exception_class("ConfigEntryAuthFailed")
_exc.ConfigEntryNotReady = _exception_class("ConfigEntryNotReady")

_dev = sys.modules["homeassistant.helpers.device_registry"]
_dev.DeviceEntryType = _empty_class("DeviceEntryType")
_dev.DeviceInfo = _empty_class("DeviceInfo")

sys.modules["homeassistant.data_entry_flow"].FlowResult = _empty_class("FlowResult")

_const = sys.modules["homeassistant.const"]
_const.CONF_API_KEY = "api_key"
_const.CONF_LLM_HASS_API = "llm_hass_api"
_const.MATCH_ALL = "*"


# ToolInput needs to round-trip its constructor args so tests can inspect them.
class _ToolInput:
    def __init__(self, id: str, tool_name: str, tool_args: dict) -> None:
        self.id = id
        self.tool_name = tool_name
        self.tool_args = tool_args

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _ToolInput)
            and self.id == other.id
            and self.tool_name == other.tool_name
            and self.tool_args == other.tool_args
        )

    def __repr__(self) -> str:
        return f"ToolInput(id={self.id!r}, tool_name={self.tool_name!r}, tool_args={self.tool_args!r})"


sys.modules["homeassistant.helpers.llm"].ToolInput = _ToolInput


# SelectOptionDict is used as a real data-holding container
sys.modules["homeassistant.helpers.selector"].SelectOptionDict = type(
    "SelectOptionDict", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}
)


# aiohttp needs real exception types so ``except aiohttp.ClientError`` works
_aiohttp = sys.modules["aiohttp"]
_aiohttp.ClientError = type("ClientError", (Exception,), {})
_aiohttp.ClientTimeout = type("ClientTimeout", (), {"__init__": lambda self, **kw: None})

# ---------------------------------------------------------------------------
# Wire submodules as attributes of their parent modules.
#
# ``MagicMock`` auto-creates attributes on access, so without explicit wiring
# ``from homeassistant.helpers import llm`` would bind ``llm`` to a fresh
# auto-Mock attribute of ``helpers`` rather than to our registered
# ``sys.modules["homeassistant.helpers.llm"]`` (which has the real ToolInput).
# Set the attribute on the parent so attribute lookup hits our submodule.
# ---------------------------------------------------------------------------

_HIERARCHY = {
    "homeassistant": [
        "components", "config_entries", "const", "core",
        "data_entry_flow", "exceptions", "helpers",
    ],
    "homeassistant.components": ["conversation", "stt", "tts"],
    "homeassistant.helpers": [
        "aiohttp_client", "config_validation", "device_registry",
        "entity_platform", "intent", "llm", "selector", "typing",
    ],
}
for _parent, _children in _HIERARCHY.items():
    _parent_mod = sys.modules[_parent]
    for _child in _children:
        setattr(_parent_mod, _child, sys.modules[f"{_parent}.{_child}"])


# ---------------------------------------------------------------------------
# Path setup — make ``custom_components/`` discoverable as a top-level path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CC_PATH = _REPO_ROOT / "custom_components"
if str(_CC_PATH) not in sys.path:
    sys.path.insert(0, str(_CC_PATH))
