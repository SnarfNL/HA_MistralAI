"""Constants for the Mistral AI Conversation integration."""

DOMAIN = "mistral_conversation"

# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_CONTINUE_CONVERSATION = "continue_conversation"
CONF_WEB_SEARCH = "web_search"
CONF_STT_LANGUAGE = "stt_language"
CONF_TTS_VOICE = "tts_voice"
# Note: device control uses HA's native CONF_LLM_HASS_API from homeassistant.const

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "ministral-8b-latest"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7          # Mistral range: 0.0–1.0
DEFAULT_CONTINUE_CONVERSATION = False
DEFAULT_WEB_SEARCH = False
DEFAULT_STT_LANGUAGE = ""          # empty = Voxtral auto-detect
DEFAULT_TTS_VOICE = "en_paul_neutral"

DEFAULT_PROMPT = (
    "You are a helpful voice assistant for a smart home called {{ ha_name }}.\n"
    "Answer in the same language the user speaks.\n"
    "Be concise and friendly.\n"
    "Today is {{ now().strftime('%A, %B %d, %Y') }}."
)

# ---------------------------------------------------------------------------
# Available chat models
# Ordered by suitability for home automation (fast + instruction-following first)
# ---------------------------------------------------------------------------
CHAT_MODELS = [
    "ministral-8b-latest",    # Best for HA: fast, great instruction following, low cost
    "ministral-3b-latest",    # Ultra-fast, lightweight, simple commands
    "mistral-small-latest",   # Balanced: speed + quality
    "mistral-medium-latest",  # Required for web search via Agents API
    "mistral-large-latest",   # Most capable, best for complex reasoning
    "open-mistral-nemo",      # Open-source, compact
]

# Models that support the Agents/Conversations API (required for web search)
AGENT_CAPABLE_MODELS = [
    "mistral-medium-latest",
    "mistral-medium-2505",
    "mistral-large-latest",
]

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
STT_MODEL = "voxtral-mini-latest"

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
TTS_MODEL = "voxtral-mini-tts-2603"

# Available voices per Mistral TTS documentation (voxtral-mini-tts-2603)
TTS_VOICES = [
    "en_paul_angry",
    "en_paul_cheerful",
    "en_paul_confident",
    "en_paul_excited",
    "en_paul_frustrated",
    "en_paul_happy",
    "en_paul_neutral",
    "en_paul_sad",
    "fr_marie_angry",
    "fr_marie_curious",
    "fr_marie_excited",
    "fr_marie_happy",
    "fr_marie_neutral",
    "fr_marie_sad",
    "gb_jane_confused",
    "gb_jane_curious",
    "gb_jane_frustrated",
    "gb_jane_jealousy",
    "gb_jane_neutral",
    "gb_jane_sad",
    "gb_jane_sarcasm",
    "gb_jane_shameful",
    "gb_oliver_angry",
    "gb_oliver_cheerful",
    "gb_oliver_confident",
    "gb_oliver_curious",
    "gb_oliver_excited",
    "gb_oliver_neutral",
    "gb_oliver_sad"
]

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
MISTRAL_API_BASE = "https://api.mistral.ai/v1"

# Max tool-call round-trips to prevent infinite loops
MAX_TOOL_ITERATIONS = 10
