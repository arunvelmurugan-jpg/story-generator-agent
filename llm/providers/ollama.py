"""
Ollama LLM Provider - Local Model Inference

Supports local LLM inference using Ollama.
Equivalent to n8n's Ollama Chat Model node.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama provider."""
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 4096
    num_ctx: int = 4096
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    timeout: int = 120


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local model inference."""
    
    SUPPORTED_MODELS = [
        "llama2", "llama3", "mistral", "mixtral", "codellama",
        "phi", "phi3", "gemma", "qwen", "deepseek", "neural-chat"
    ]
    
    def __init__(self, config: Optional[OllamaConfig] = None, **kwargs):
        self.config = config or OllamaConfig(**kwargs)
        self._async_client = None
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    async def _get_client(self):
        if self._async_client is None:
            try:
                import httpx
                self._async_client = httpx.AsyncClient(
                    base_url=self.config.base_url, timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("httpx required: pip install httpx")
        return self._async_client
    
    async def complete(
        self, messages: List[Message], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> LLMResponse:
        client = await self._get_client()
        payload = {
            "model": model or self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.num_predict
            }
        }
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", model or self.config.model),
            provider=self.provider_name,
            usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                   "completion_tokens": data.get("eval_count", 0)}
        )
    
    async def stream(
        self, messages: List[Message], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = await self._get_client()
        payload = {
            "model": model or self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True
        }
        async with client.stream("POST", "/api/chat", json=payload) as response:
            import json as json_lib
            async for line in response.aiter_lines():
                if line:
                    data = json_lib.loads(line)
                    yield StreamChunk(
                        content=data.get("message", {}).get("content", ""),
                        done=data.get("done", False)
                    )
    
    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False


__all__ = ["OllamaProvider", "OllamaConfig"]
