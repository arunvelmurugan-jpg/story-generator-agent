"""
Mistral AI LLM Provider
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import os

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class MistralConfig:
    api_key: Optional[str] = None
    model: str = "mistral-large-latest"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    safe_prompt: bool = False


class MistralProvider(BaseLLMProvider):
    """Mistral AI LLM provider."""
    
    SUPPORTED_MODELS = [
        "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
        "codestral-latest", "open-mistral-7b", "open-mixtral-8x7b", "mistral-embed"
    ]
    
    def __init__(self, config: Optional[MistralConfig] = None, **kwargs):
        self.config = config or MistralConfig(**kwargs)
        if not self.config.api_key:
            self.config.api_key = os.environ.get("MISTRAL_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "mistral"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    def _get_client(self):
        if self._client is None:
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("mistralai required: pip install mistralai")
        return self._client
    
    async def complete(
        self, messages: List[Message], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> LLMResponse:
        client = self._get_client()
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        response = await client.chat.complete_async(
            model=model or self.config.model, messages=formatted,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model, provider=self.provider_name,
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens}
        )
    
    async def stream(
        self, messages: List[Message], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        stream = await client.chat.stream_async(
            model=model or self.config.model, messages=formatted,
            temperature=temperature or self.config.temperature
        )
        async for chunk in stream:
            if chunk.data.choices[0].delta.content:
                yield StreamChunk(content=chunk.data.choices[0].delta.content, done=False)
        yield StreamChunk(content="", done=True)
    
    async def generate_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        client = self._get_client()
        response = await client.embeddings.create_async(model=model or "mistral-embed", inputs=texts)
        return [item.embedding for item in response.data]
    
    async def health_check(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False


__all__ = ["MistralProvider", "MistralConfig"]
