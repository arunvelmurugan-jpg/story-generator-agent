"""
Base LLM Provider for PHTN.AI Sub-Agent Framework

Abstract base class for all LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Standardized chat message used by LLM providers."""
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    role: str = "assistant"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement:
    - complete(): Synchronous completion
    - stream(): Streaming completion
    - health_check(): Provider health check
    """
    
    provider_name: str = "base"
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize provider.
        
        Args:
            model: Model identifier
            api_key: API key (optional, can use env vars)
            base_url: Custom base URL (optional)
            **kwargs: Additional provider-specific config
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
        
        self._client = None
        self._initialized = False
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion.
        
        Args:
            messages: Chat messages
            tools: Tool definitions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate streaming completion.
        
        Args:
            messages: Chat messages
            tools: Tool definitions (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            StreamChunk
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health.
        
        Returns:
            Health status dictionary
        """
        pass
    
    async def close(self):
        """Close provider connections."""
        self._client = None
        self._initialized = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
        }
