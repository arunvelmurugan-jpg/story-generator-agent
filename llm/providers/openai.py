"""
OpenAI Provider for PHTN.AI Sub-Agent Framework

Comprehensive OpenAI SDK support including:
- Chat Completions API (GPT-4, GPT-4o, o1, etc.)
- Azure OpenAI Service
- Structured Outputs (JSON Schema)
- Vision/Image inputs (GPT-4V, GPT-4o)
- Audio inputs/outputs (GPT-4o-audio)
- Streaming with tool calls
- Function/Tool calling
- OpenAI-compatible APIs (Ollama, vLLM, etc.)
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator, Union

from ..base import BaseLLMProvider, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)


class OpenAIModel(str, Enum):
    """OpenAI model identifiers."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_AUDIO = "gpt-4o-audio-preview"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"


class ResponseFormat(str, Enum):
    """Response format types."""
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass
class ImageContent:
    """Image content for vision models."""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    media_type: str = "image/png"
    detail: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        if self.url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": self.url,
                    "detail": self.detail,
                },
            }
        elif self.base64_data:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.media_type};base64,{self.base64_data}",
                    "detail": self.detail,
                },
            }
        return {}


@dataclass
class AudioContent:
    """Audio content for audio models."""
    data: str
    format: str = "wav"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "input_audio",
            "input_audio": {
                "data": self.data,
                "format": self.format,
            },
        }


@dataclass
class StructuredOutput:
    """Structured output configuration."""
    name: str
    schema: Dict[str, Any]
    strict: bool = True
    description: Optional[str] = None
    
    def to_response_format(self) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.schema,
                "strict": self.strict,
            },
        }


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider with full SDK support.
    
    Supports:
    - OpenAI API (GPT-4, GPT-4o, o1, etc.)
    - Azure OpenAI Service
    - OpenAI-compatible APIs (Ollama, vLLM, LiteLLM)
    - Vision/Image inputs
    - Audio inputs/outputs
    - Structured outputs (JSON Schema)
    - Function/Tool calling
    - Streaming
    """
    
    provider_name = "openai"
    
    VISION_MODELS = {
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview",
        "gpt-4-vision-preview", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06",
    }
    
    AUDIO_MODELS = {
        "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-10-01",
    }
    
    STRUCTURED_OUTPUT_MODELS = {
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
    }
    
    O1_MODELS = {
        "o1", "o1-mini", "o1-preview",
    }

    # gpt-5 family also requires `max_completion_tokens` (not `max_tokens`).
    GPT5_MODELS = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-turbo",
    }
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-15-preview",
        azure_deployment: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (or Azure API key)
            base_url: Custom base URL for OpenAI-compatible APIs
            organization: OpenAI organization ID
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name
            default_headers: Default headers for requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(model, api_key, base_url, **kwargs)
        self.organization = organization
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._is_azure = bool(azure_endpoint)
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                if self._is_azure:
                    from openai import AsyncAzureOpenAI
                    
                    self._client = AsyncAzureOpenAI(
                        api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                        azure_endpoint=self.azure_endpoint,
                        api_version=self.azure_api_version,
                        default_headers=self.default_headers,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
                else:
                    from openai import AsyncOpenAI
                    
                    self._client = AsyncOpenAI(
                        api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                        base_url=self.base_url,
                        organization=self.organization,
                        default_headers=self.default_headers,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")
        
        return self._client
    
    @property
    def supports_vision(self) -> bool:
        """Check if model supports vision/images."""
        return any(v in self.model for v in self.VISION_MODELS)
    
    @property
    def supports_audio(self) -> bool:
        """Check if model supports audio."""
        return any(a in self.model for a in self.AUDIO_MODELS)
    
    @property
    def supports_structured_output(self) -> bool:
        """Check if model supports structured outputs."""
        return any(s in self.model for s in self.STRUCTURED_OUTPUT_MODELS)
    
    @property
    def is_o1_model(self) -> bool:
        """Check if model is an o1 reasoning model."""
        return any(o in self.model for o in self.O1_MODELS)

    @property
    def is_gpt5_model(self) -> bool:
        """Check if model is a gpt-5 family model."""
        return any(g in self.model for g in self.GPT5_MODELS)

    @property
    def uses_max_completion_tokens(self) -> bool:
        """OpenAI deprecated `max_tokens` for o1 + gpt-5 families; both must
        send `max_completion_tokens` instead."""
        return self.is_o1_model or self.is_gpt5_model

    # Params OpenAI rejects on o1 + gpt-5 reasoning models. Sending any of
    # these returns HTTP 400 ("Unsupported parameter" or "Unsupported value:
    # X does not support Y with this model"). Stripping them defensively
    # before the API call is safer than scattering conditionals.
    _REASONING_UNSUPPORTED_PARAMS = (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "max_tokens",  # already swapped to max_completion_tokens upstream
        "n",
    )

    def _sanitize_for_reasoning_model(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Strip params unsupported by o1/gpt-5 from a chat-completions request,
        and inject reasoning_effort=minimal for gpt-5 so the model returns text
        content instead of burning its output budget on reasoning tokens."""
        if not self.uses_max_completion_tokens:
            return request_params
        for key in self._REASONING_UNSUPPORTED_PARAMS:
            request_params.pop(key, None)
        # gpt-5 supports reasoning_effort: minimal|low|medium|high (default medium).
        # On medium/high, the model can consume the entire completion budget on
        # internal reasoning and return message.content=None — which breaks all
        # downstream callers that read the text. "minimal" forces it to produce
        # visible output quickly (similar latency to gpt-4o-mini).
        if self.is_gpt5_model and "reasoning_effort" not in request_params:
            request_params["reasoning_effort"] = "minimal"
        return request_params
    
    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[ImageContent]] = None,
        audio: Optional[List[AudioContent]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build messages with multimodal content.
        
        Args:
            messages: Base messages
            images: Optional image content
            audio: Optional audio content
            
        Returns:
            Formatted messages
        """
        if not images and not audio:
            return messages
        
        formatted = []
        for msg in messages:
            if msg.get("role") == "user" and (images or audio):
                content = []
                
                if isinstance(msg.get("content"), str):
                    content.append({"type": "text", "text": msg["content"]})
                elif isinstance(msg.get("content"), list):
                    content.extend(msg["content"])
                
                if images:
                    for img in images:
                        content.append(img.to_dict())
                
                if audio:
                    for aud in audio:
                        content.append(aud.to_dict())
                
                formatted.append({
                    "role": msg["role"],
                    "content": content,
                })
            else:
                formatted.append(msg)
        
        return formatted
    
    def _build_response_format(
        self,
        response_format: Optional[Union[str, Dict[str, Any], StructuredOutput]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build response format configuration."""
        if response_format is None:
            return None
        
        if isinstance(response_format, StructuredOutput):
            return response_format.to_response_format()
        
        if isinstance(response_format, str):
            if response_format == "json_object":
                return {"type": "json_object"}
            elif response_format == "text":
                return {"type": "text"}
            return None
        
        if isinstance(response_format, dict):
            return response_format
        
        return None
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[List[ImageContent]] = None,
        audio: Optional[List[AudioContent]] = None,
        response_format: Optional[Union[str, Dict[str, Any], StructuredOutput]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion using OpenAI API.
        
        Args:
            messages: Chat messages
            tools: Tool definitions for function calling
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            images: Image content for vision models
            audio: Audio content for audio models
            response_format: Response format (text, json_object, or StructuredOutput)
            **kwargs: Additional parameters (top_p, seed, stop, etc.)
            
        Returns:
            LLMResponse with completion
        """
        client = self._get_client()
        start_time = time.time()
        
        formatted_messages = self._build_messages(messages, images, audio)
        
        request_params = {
            "model": self.azure_deployment if self._is_azure else self.model,
            "messages": formatted_messages,
        }
        
        if not self.uses_max_completion_tokens:
            request_params["temperature"] = temperature
        
        if max_tokens:
            if self.uses_max_completion_tokens:
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
        
        if tools and not self.is_o1_model:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.pop("tool_choice", "auto")
            
            if kwargs.get("parallel_tool_calls") is not None:
                request_params["parallel_tool_calls"] = kwargs.pop("parallel_tool_calls")
        
        fmt = self._build_response_format(response_format)
        if fmt and not self.is_o1_model:
            request_params["response_format"] = fmt
        
        if self.supports_audio and audio:
            request_params["modalities"] = ["text", "audio"]
            request_params["audio"] = {
                "voice": kwargs.pop("voice", "alloy"),
                "format": kwargs.pop("audio_format", "wav"),
            }
        
        for key in ["top_p", "presence_penalty", "frequency_penalty", "seed", "stop", "logprobs", "top_logprobs", "user"]:
            if key in kwargs and not self.is_o1_model:
                request_params[key] = kwargs.pop(key)
        
        request_params = self._sanitize_for_reasoning_model(request_params)
        response = await client.chat.completions.create(**request_params)
        
        latency_ms = (time.time() - start_time) * 1000
        
        message = response.choices[0].message
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        
        audio_response = None
        if hasattr(message, 'audio') and message.audio:
            audio_response = {
                "id": message.audio.id,
                "data": message.audio.data,
                "transcript": message.audio.transcript,
                "expires_at": message.audio.expires_at,
            }
        
        parsed_content = None
        if response_format and isinstance(response_format, StructuredOutput):
            try:
                if hasattr(message, 'parsed') and message.parsed:
                    parsed_content = message.parsed
                elif message.content:
                    parsed_content = json.loads(message.content)
            except json.JSONDecodeError:
                pass
        
        return LLMResponse(
            content=message.content or "",
            role=message.role,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms,
            metadata={
                "id": response.id,
                "created": response.created,
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "audio": audio_response,
                "parsed": parsed_content,
            },
        )
    
    async def complete_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        output_schema: StructuredOutput,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with structured output (JSON Schema).
        
        Args:
            messages: Chat messages
            output_schema: StructuredOutput configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Parsed JSON response matching the schema
        """
        if not self.supports_structured_output:
            logger.warning(f"Model {self.model} may not support structured outputs")
        
        response = await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=output_schema,
            **kwargs,
        )
        
        if response.metadata and response.metadata.get("parsed"):
            return response.metadata["parsed"]
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"raw_content": response.content}
    
    async def complete_with_vision(
        self,
        messages: List[Dict[str, Any]],
        images: List[Union[str, ImageContent]],
        detail: str = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion with image inputs.
        
        Args:
            messages: Chat messages
            images: List of image URLs or ImageContent objects
            detail: Image detail level ("low", "high", "auto")
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            LLMResponse with completion
        """
        if not self.supports_vision:
            raise ValueError(f"Model {self.model} does not support vision")
        
        image_contents = []
        for img in images:
            if isinstance(img, str):
                if img.startswith("data:"):
                    parts = img.split(",", 1)
                    media_type = parts[0].split(";")[0].replace("data:", "")
                    image_contents.append(ImageContent(
                        base64_data=parts[1] if len(parts) > 1 else "",
                        media_type=media_type,
                        detail=detail,
                    ))
                elif img.startswith("http"):
                    image_contents.append(ImageContent(url=img, detail=detail))
                else:
                    image_contents.append(ImageContent(base64_data=img, detail=detail))
            elif isinstance(img, ImageContent):
                image_contents.append(img)
        
        return await self.complete(
            messages=messages,
            images=image_contents,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    async def complete_with_audio(
        self,
        messages: List[Dict[str, Any]],
        audio_data: Union[str, bytes],
        audio_format: str = "wav",
        voice: str = "alloy",
        output_format: str = "wav",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion with audio input/output.
        
        Args:
            messages: Chat messages
            audio_data: Base64 audio data or bytes
            audio_format: Input audio format (wav, mp3, etc.)
            voice: Output voice (alloy, echo, fable, onyx, nova, shimmer)
            output_format: Output audio format
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            LLMResponse with audio in metadata
        """
        if not self.supports_audio:
            raise ValueError(f"Model {self.model} does not support audio")
        
        if isinstance(audio_data, bytes):
            audio_data = base64.b64encode(audio_data).decode()
        
        audio_content = [AudioContent(data=audio_data, format=audio_format)]
        
        return await self.complete(
            messages=messages,
            audio=audio_content,
            temperature=temperature,
            max_tokens=max_tokens,
            voice=voice,
            audio_format=output_format,
            **kwargs,
        )
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[List[ImageContent]] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate streaming completion.
        
        Args:
            messages: Chat messages
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            images: Image content for vision
            response_format: Response format
            
        Yields:
            StreamChunk with content deltas
        """
        if self.is_o1_model:
            response = await self.complete(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            yield StreamChunk(
                content=response.content,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )
            return
        
        client = self._get_client()
        
        formatted_messages = self._build_messages(messages, images)
        
        request_params = {
            "model": self.azure_deployment if self._is_azure else self.model,
            "messages": formatted_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not self.uses_max_completion_tokens:
            request_params["temperature"] = temperature
        
        if max_tokens:
            if self.uses_max_completion_tokens:
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
        
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        fmt = self._build_response_format(response_format)
        if fmt:
            request_params["response_format"] = fmt
        
        for key in ["top_p", "presence_penalty", "frequency_penalty", "seed", "stop"]:
            if key in kwargs:
                request_params[key] = kwargs[key]
        
        request_params = self._sanitize_for_reasoning_model(request_params)
        stream = await client.chat.completions.create(**request_params)
        
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        
        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    yield StreamChunk(
                        content="",
                        usage={
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        },
                    )
                continue
            
            delta = chunk.choices[0].delta
            
            tool_calls = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    
                    if tc.id:
                        accumulated_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                    
                    tool_calls.append({
                        "index": idx,
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else None,
                        },
                    })
            
            finish_reason = chunk.choices[0].finish_reason
            
            if finish_reason == "tool_calls" and accumulated_tool_calls:
                yield StreamChunk(
                    content=delta.content or "",
                    tool_calls=list(accumulated_tool_calls.values()),
                    finish_reason=finish_reason,
                )
            else:
                yield StreamChunk(
                    content=delta.content or "",
                    tool_calls=tool_calls if tool_calls else None,
                    finish_reason=finish_reason,
                )
    
    async def create_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model
            dimensions: Output dimensions (for ada-3 models)
            
        Returns:
            List of embedding vectors
        """
        client = self._get_client()
        
        request_params = {
            "model": model,
            "input": texts,
        }
        
        if dimensions:
            request_params["dimensions"] = dimensions
        
        response = await client.embeddings.create(**request_params)
        
        return [item.embedding for item in response.data]
    
    async def moderate(
        self,
        text: str,
        model: str = "text-moderation-latest",
    ) -> Dict[str, Any]:
        """
        Check text for policy violations using OpenAI Moderation API.
        
        Args:
            text: Text to moderate
            model: Moderation model
            
        Returns:
            Moderation results with categories and scores
        """
        client = self._get_client()
        
        response = await client.moderations.create(
            model=model,
            input=text,
        )
        
        result = response.results[0]
        
        return {
            "flagged": result.flagged,
            "categories": {
                "hate": result.categories.hate,
                "hate_threatening": result.categories.hate_threatening,
                "harassment": result.categories.harassment,
                "harassment_threatening": result.categories.harassment_threatening,
                "self_harm": result.categories.self_harm,
                "self_harm_intent": result.categories.self_harm_intent,
                "self_harm_instructions": result.categories.self_harm_instructions,
                "sexual": result.categories.sexual,
                "sexual_minors": result.categories.sexual_minors,
                "violence": result.categories.violence,
                "violence_graphic": result.categories.violence_graphic,
            },
            "category_scores": {
                "hate": result.category_scores.hate,
                "hate_threatening": result.category_scores.hate_threatening,
                "harassment": result.category_scores.harassment,
                "harassment_threatening": result.category_scores.harassment_threatening,
                "self_harm": result.category_scores.self_harm,
                "self_harm_intent": result.category_scores.self_harm_intent,
                "self_harm_instructions": result.category_scores.self_harm_instructions,
                "sexual": result.category_scores.sexual,
                "sexual_minors": result.category_scores.sexual_minors,
                "violence": result.category_scores.violence,
                "violence_graphic": result.category_scores.violence_graphic,
            },
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            client = self._get_client()
            
            test_model = self.model
            if self.is_o1_model:
                test_model = "gpt-4o-mini"
            
            response = await client.chat.completions.create(
                model=self.azure_deployment if self._is_azure else test_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "is_azure": self._is_azure,
                "supports_vision": self.supports_vision,
                "supports_audio": self.supports_audio,
                "supports_structured_output": self.supports_structured_output,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "is_azure": self._is_azure,
                "error": str(e),
            }


class AzureOpenAIProvider(OpenAIProvider):
    """
    Azure OpenAI provider.
    
    Convenience class for Azure OpenAI with simplified configuration.
    """
    
    provider_name = "azure_openai"
    
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        **kwargs,
    ):
        """
        Initialize Azure OpenAI provider.
        
        Args:
            deployment: Azure deployment name
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure API key
            api_version: Azure API version
        """
        super().__init__(
            model=deployment,
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_api_version=api_version,
            azure_deployment=deployment,
            **kwargs,
        )
