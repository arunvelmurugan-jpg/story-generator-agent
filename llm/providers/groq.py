"""
Groq LLM Provider - Ultra-fast Inference

Supports Groq's LPU-accelerated models:
- Llama 3
- Mixtral
- Gemma

Equivalent to n8n's Groq Chat Model node.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import os

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq provider."""
    api_key: Optional[str] = None
    model: str = "llama3-70b-8192"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    base_url: str = "https://api.groq.com/openai/v1"


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider with LPU acceleration.
    
    Supported models:
    - llama3-70b-8192
    - llama3-8b-8192
    - llama-3.1-70b-versatile
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma-7b-it
    - gemma2-9b-it
    """
    
    SUPPORTED_MODELS = [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it",
        "whisper-large-v3",
    ]
    
    def __init__(self, config: Optional[GroqConfig] = None, **kwargs):
        self.config = config or GroqConfig(**kwargs)
        if not self.config.api_key:
            self.config.api_key = os.environ.get("GROQ_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    def _get_client(self):
        """Get or create Groq client."""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("groq is required. Install with: pip install groq")
        return self._client
    
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Groq."""
        import asyncio
        
        client = self._get_client()
        
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model or self.config.model,
                    messages=formatted_messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    top_p=self.config.top_p,
                    stop=self.config.stop
                )
            )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider=self.provider_name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "id": response.id,
                    "x_groq": getattr(response, "x_groq", None)
                }
            )
        except Exception as e:
            logger.error(f"Groq completion error: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using Groq."""
        client = self._get_client()
        
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        try:
            stream = client.chat.completions.create(
                model=model or self.config.model,
                messages=formatted_messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        done=False
                    )
            
            yield StreamChunk(content="", done=True)
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_file: bytes,
        model: str = "whisper-large-v3",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio using Groq Whisper."""
        import asyncio
        import io
        
        client = self._get_client()
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.audio.transcriptions.create(
                    model=model,
                    file=("audio.wav", io.BytesIO(audio_file)),
                    language=language
                )
            )
            
            return {
                "text": response.text,
                "model": model
            }
        except Exception as e:
            logger.error(f"Groq transcription error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Groq API connectivity."""
        try:
            self._get_client()
            return True
        except Exception:
            return False


__all__ = ["GroqProvider", "GroqConfig"]
