"""
LLM Abstraction Layer for PHTN.AI Sub-Agent Framework

Provides a unified interface for multiple LLM providers with:
- Multi-provider support (OpenAI, Anthropic, Azure, Bedrock, etc.)
- Intelligent routing and fallbacks
- Circuit breaker pattern
- Token tracking and cost management
- Vision/Image support (GPT-4V, GPT-4o)
- Audio support (GPT-4o-audio)
- Structured outputs (JSON Schema)
- Embeddings and Moderation APIs
"""

from .router import LLMRouter
from .base import BaseLLMProvider, LLMResponse, StreamChunk
from .providers.openai import (
    OpenAIProvider,
    AzureOpenAIProvider,
    OpenAIModel,
    ResponseFormat,
    ImageContent,
    AudioContent,
    StructuredOutput,
)
from .providers.anthropic import AnthropicProvider
from .providers.mock import MockLLMProvider

__all__ = [
    "LLMRouter",
    "BaseLLMProvider",
    "LLMResponse",
    "StreamChunk",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "OpenAIModel",
    "ResponseFormat",
    "ImageContent",
    "AudioContent",
    "StructuredOutput",
    "AnthropicProvider",
    "MockLLMProvider",
]
