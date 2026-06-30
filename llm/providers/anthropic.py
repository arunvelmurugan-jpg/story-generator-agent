"""
Anthropic Provider for PHTN.AI Sub-Agent Framework

Supports Claude models via Anthropic API.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, AsyncIterator

from ..base import BaseLLMProvider, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM provider for Claude models.
    
    Supports:
    - Claude 3.5 Sonnet
    - Claude 3 Opus
    - Claude 3 Haiku
    """
    
    provider_name = "anthropic"
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, api_key, base_url, **kwargs)
        self._client = None
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                
                self._client = AsyncAnthropic(
                    api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        client = self._get_client()
        start_time = time.time()
        
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        request_params = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        if tools:
            request_params["tools"] = self._convert_tools(tools)
        
        response = await client.messages.create(**request_params)
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })
        
        return LLMResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
        )
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion."""
        client = self._get_client()
        
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        request_params = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        async with client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text)
            
            final_message = await stream.get_final_message()
            yield StreamChunk(
                content="",
                finish_reason=final_message.stop_reason,
                usage={
                    "prompt_tokens": final_message.usage.input_tokens,
                    "completion_tokens": final_message.usage.output_tokens,
                },
            )
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        
        return anthropic_tools
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health."""
        try:
            client = self._get_client()
            
            response = await client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e),
            }
