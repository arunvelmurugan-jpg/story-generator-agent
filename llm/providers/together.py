"""
Together AI LLM Provider
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import os

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class TogetherConfig:
    api_key: Optional[str] = None
    model: str = "meta-llama/Llama-3-70b-chat-hf"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.7
    top_k: int = 50


class TogetherProvider(BaseLLMProvider):
    """Together AI LLM provider with 100+ open-source models."""
    
    SUPPORTED_MODELS = [
        "meta-llama/Llama-3-70b-chat-hf", "meta-llama/Llama-3-8b-chat-hf",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1", "Qwen/Qwen2-72B-Instruct",
        "deepseek-ai/deepseek-coder-33b-instruct"
    ]
    
    def __init__(self, config: Optional[TogetherConfig] = None, **kwargs):
        self.config = config or TogetherConfig(**kwargs)
        if not self.config.api_key:
            self.config.api_key = os.environ.get("TOGETHER_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "together"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    def _get_client(self):
        if self._client is None:
            try:
                from together import Together
                self._client = Together(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("together required: pip install together")
        return self._client
    
    async def complete(
        self, messages: List[Message], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> LLMResponse:
        import asyncio
        client = self._get_client()
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.chat.completions.create(
                model=model or self.config.model, messages=formatted,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
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
        stream = client.chat.completions.create(
            model=model or self.config.model, messages=formatted,
            temperature=temperature or self.config.temperature, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content, done=False)
        yield StreamChunk(content="", done=True)
    
    async def health_check(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False


__all__ = ["TogetherProvider", "TogetherConfig"]
