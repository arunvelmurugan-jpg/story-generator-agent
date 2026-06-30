"""LLM Providers Module -- lazy / optional imports.

OpenAI is mandatory in the production deployment (see requirements.txt).
Every other provider is optional -- its underlying SDK may not be installed.
We import each one inside a try/except so a missing SDK does NOT crash the
entire LLM module (which would prevent every sub-agent that uses an LLM
from running, manifesting as "Stream 1/2/3 ... done with no data").

Providers that fail to import are silently omitted from ``__all__``; the
router checks availability before instantiating.
"""

# Mandatory -- production always has openai installed.
from .openai import OpenAIProvider, AzureOpenAIProvider

__all__ = ["OpenAIProvider", "AzureOpenAIProvider"]

_OPTIONAL_PROVIDERS = [
    ("anthropic", "AnthropicProvider"),
    ("ollama",    "OllamaProvider"),
    ("bedrock",   "BedrockProvider"),
    ("gemini",    "GeminiProvider"),
    ("groq",      "GroqProvider"),
    ("mistral",   "MistralProvider"),
    ("together",  "TogetherProvider"),
    ("cohere",    "CohereProvider"),
]

for _module_name, _class_name in _OPTIONAL_PROVIDERS:
    try:
        _mod = __import__(f"{__name__}.{_module_name}", fromlist=[_class_name])
        globals()[_class_name] = getattr(_mod, _class_name)
        __all__.append(_class_name)
    except Exception:  # SDK missing or its own imports fail -- non-fatal
        pass
