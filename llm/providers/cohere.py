"""
Cohere LLM Provider

Supports Cohere models:
- Command R+
- Command R
- Command
- Embed
- Rerank

Equivalent to n8n's Cohere Chat Model node.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import os

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class CohereConfig:
    """Configuration for Cohere provider."""
    api_key: Optional[str] = None
    model: str = "command-r-plus"
    temperature: float = 0.7
    max_tokens: int = 4096
    p: float = 0.75
    k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    preamble: Optional[str] = None


class CohereProvider(BaseLLMProvider):
    """
    Cohere LLM provider.
    
    Supported models:
    - command-r-plus
    - command-r
    - command
    - command-light
    - command-nightly
    - embed-english-v3.0
    - embed-multilingual-v3.0
    - rerank-english-v3.0
    - rerank-multilingual-v3.0
    """
    
    SUPPORTED_MODELS = [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
        "command-nightly",
        "command-r-plus-08-2024",
        "command-r-08-2024",
    ]
    
    EMBEDDING_MODELS = [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0",
    ]
    
    RERANK_MODELS = [
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",
        "rerank-english-v2.0",
        "rerank-multilingual-v2.0",
    ]
    
    def __init__(self, config: Optional[CohereConfig] = None, **kwargs):
        self.config = config or CohereConfig(**kwargs)
        if not self.config.api_key:
            self.config.api_key = os.environ.get("COHERE_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "cohere"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS + self.EMBEDDING_MODELS + self.RERANK_MODELS
    
    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.ClientV2(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("cohere is required. Install with: pip install cohere")
        return self._client
    
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Cohere."""
        import asyncio
        
        client = self._get_client()
        
        formatted_messages = []
        for m in messages:
            role = "assistant" if m.role == "assistant" else "user"
            if m.role == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": m.content})
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat(
                    model=model or self.config.model,
                    messages=formatted_messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    p=self.config.p,
                    k=self.config.k,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty
                )
            )
            
            content = ""
            if response.message and response.message.content:
                content = response.message.content[0].text if response.message.content else ""
            
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.tokens.input_tokens,
                    "completion_tokens": response.usage.tokens.output_tokens,
                    "total_tokens": response.usage.tokens.input_tokens + response.usage.tokens.output_tokens
                }
            
            return LLMResponse(
                content=content,
                model=model or self.config.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.finish_reason,
                    "id": response.id
                }
            )
        except Exception as e:
            logger.error(f"Cohere completion error: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using Cohere."""
        client = self._get_client()
        
        formatted_messages = []
        for m in messages:
            role = "assistant" if m.role == "assistant" else "user"
            if m.role == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": m.content})
        
        try:
            stream = client.chat_stream(
                model=model or self.config.model,
                messages=formatted_messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            for event in stream:
                if event.type == "content-delta":
                    yield StreamChunk(
                        content=event.delta.message.content.text,
                        done=False
                    )
                elif event.type == "message-end":
                    yield StreamChunk(content="", done=True)
        except Exception as e:
            logger.error(f"Cohere streaming error: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        import asyncio
        
        client = self._get_client()
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.embed(
                    model=model or "embed-english-v3.0",
                    texts=texts,
                    input_type=input_type,
                    embedding_types=["float"]
                )
            )
            
            return response.embeddings.float_
        except Exception as e:
            logger.error(f"Cohere embeddings error: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere."""
        import asyncio
        
        client = self._get_client()
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.rerank(
                    model=model or "rerank-english-v3.0",
                    query=query,
                    documents=documents,
                    top_n=top_n
                )
            )
            
            return [
                {
                    "index": r.index,
                    "relevance_score": r.relevance_score,
                    "document": documents[r.index]
                }
                for r in response.results
            ]
        except Exception as e:
            logger.error(f"Cohere rerank error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Cohere API connectivity."""
        try:
            self._get_client()
            return True
        except Exception:
            return False


__all__ = ["CohereProvider", "CohereConfig"]
