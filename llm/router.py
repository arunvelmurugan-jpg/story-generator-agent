"""
LLM Router for PHTN.AI Sub-Agent Framework

Provides intelligent routing across multiple LLM providers with:
- Primary/fallback model support
- Azure OpenAI support
- Vision/Image support (GPT-4V, GPT-4o)
- Audio support (GPT-4o-audio)
- Structured outputs (JSON Schema)
- Circuit breaker pattern
- Retry with exponential backoff
- Token tracking
- Cost tracking and budget enforcement
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, AsyncIterator, Union

from .base import BaseLLMProvider, LLMResponse, StreamChunk
from ..observability.otel_logging import get_logger, log_llm_call

logger = get_logger(__name__)


# ─── Cumulative LLM usage across all router instances in this process ────────
# Surfaced via GET /api/llm-usage; reset on pod restart.
_GLOBAL_USAGE: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
_GLOBAL_COST: Dict[str, float] = {"input_usd": 0.0, "output_usd": 0.0, "total_usd": 0.0}


def get_global_usage() -> Dict[str, Any]:
    """Return cumulative token usage + USD cost since pod start."""
    return {
        "tokens": dict(_GLOBAL_USAGE),
        "cost_usd": dict(_GLOBAL_COST),
    }


def reset_global_usage() -> None:
    global _GLOBAL_USAGE, _GLOBAL_COST
    _GLOBAL_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    _GLOBAL_COST = {"input_usd": 0.0, "output_usd": 0.0, "total_usd": 0.0}



@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    failures: int = 0
    last_failure: Optional[datetime] = None
    state: str = "closed"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.utcnow()
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def is_open(self, threshold: int, cooldown_seconds: int) -> bool:
        if self.state == "open":
            if self.last_failure and \
               datetime.utcnow() - self.last_failure > timedelta(seconds=cooldown_seconds):
                self.state = "half-open"
                return False
            return True
        
        if self.failures >= threshold:
            self.state = "open"
            return True
        
        return False


class LLMRouter:
    """
    Intelligent LLM router with fallback and circuit breaker support.
    
    Features:
    - Multiple provider support
    - Automatic fallback on failure
    - Circuit breaker pattern
    - Retry with exponential backoff
    - Token usage tracking
    - Latency-aware routing (optional)
    """
    
    def __init__(
        self,
        primary_model: str,
        provider: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        routing_config: Optional[Dict[str, Any]] = None,
        azure_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        structured_output_config: Optional[Dict[str, Any]] = None,
        embeddings_config: Optional[Dict[str, Any]] = None,
        moderation_config: Optional[Dict[str, Any]] = None,
        streaming_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM Router.
        
        Args:
            primary_model: Primary model identifier
            provider: Provider name (auto-detected if not specified)
            fallback_models: List of fallback models
            parameters: Default model parameters
            routing_config: Routing configuration
            azure_config: Azure OpenAI configuration
            vision_config: Vision/Image configuration
            audio_config: Audio configuration
            structured_output_config: Structured output configuration
            embeddings_config: Embeddings configuration
            moderation_config: Moderation configuration
            streaming_config: Streaming configuration
        """
        self.primary_model = primary_model
        self.provider_name = provider or self._detect_provider(primary_model)
        self.fallback_models = fallback_models or []
        self.parameters = parameters or {}
        self.routing_config = routing_config or {}
        
        self.azure_config = azure_config or {}
        self.vision_config = vision_config or {}
        self.audio_config = audio_config or {}
        self.structured_output_config = structured_output_config or {}
        self.embeddings_config = embeddings_config or {}
        self.moderation_config = moderation_config or {}
        self.streaming_config = streaming_config or {}
        
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._token_usage: Dict[str, int] = {"input": 0, "output": 0}
        self._cost_tracking: Dict[str, float] = {"total": 0.0, "input": 0.0, "output": 0.0}
        
        self._pricing = self._get_default_pricing()
        
        self._initialize_providers()
    
    def _get_default_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get default pricing per 1K tokens."""
        return {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o-audio-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "o1": {"input": 0.015, "output": 0.06},
            "o1-mini": {"input": 0.003, "output": 0.012},
            "o1-preview": {"input": 0.015, "output": 0.06},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0},
            "default": {"input": 0.001, "output": 0.002},
        }
    
    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        model_lower = model.lower()
        
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "together"
        else:
            return "openai"
    
    def _initialize_providers(self):
        """Initialize LLM providers."""
        import os
        
        all_models = [self.primary_model] + self.fallback_models
        
        for model in all_models:
            provider_name = self._detect_provider(model)
            
            if model not in self._providers:
                provider = self._create_provider(provider_name, model)
                self._providers[model] = provider
                self._circuit_breakers[model] = CircuitBreakerState()
        
        mock_enabled = os.environ.get("PHTN_MOCK_LLM_ENABLED", "false").lower() == "true"
        if mock_enabled:
            from .providers.mock import MockLLMProvider
            mock_model = "mock-demo"
            self._providers[mock_model] = MockLLMProvider(model=mock_model)
            self._circuit_breakers[mock_model] = CircuitBreakerState()
            self._mock_model = mock_model
            logger.info("🎭 Mock LLM provider enabled as fallback")
    
    def _create_provider(self, provider_name: str, model: str) -> BaseLLMProvider:
        """Create provider instance."""
        from .providers.openai import OpenAIProvider, AzureOpenAIProvider
        from .providers.anthropic import AnthropicProvider
        
        if provider_name == "azure_openai" and self.azure_config:
            return AzureOpenAIProvider(
                deployment=self.azure_config.get("deployment", model),
                endpoint=self.azure_config.get("endpoint", ""),
                api_version=self.azure_config.get("api_version", "2024-02-15-preview"),
            )
        
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
        }
        
        provider_class = provider_map.get(provider_name, OpenAIProvider)
        return provider_class(model=model)
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic fallback.
        
        Args:
            messages: Chat messages
            tools: Tool definitions
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        merged_params = {**self.parameters, **kwargs}
        
        models_to_try = [self.primary_model] + self.fallback_models
        last_error = None
        
        for model in models_to_try:
            provider = self._providers.get(model)
            circuit_breaker = self._circuit_breakers.get(model)
            
            if not provider:
                continue
            
            cb_config = self.routing_config.get("circuit_breaker", {})
            if circuit_breaker and circuit_breaker.is_open(
                threshold=cb_config.get("failure_threshold", 5),
                cooldown_seconds=cb_config.get("cooldown_seconds", 60),
            ):
                logger.warning(f"Circuit breaker open for model: {model}")
                continue
            
            try:
                start_time = time.time()
                response = await self._complete_with_retry(
                    provider, messages, tools, merged_params
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                self._track_usage(response.usage, model)
                
                # Log LLM call for phtnai-ops-metrics FinOps tracking
                provider_name = self._detect_provider(model)
                cost = self._calculate_cost(response.usage, model)
                log_llm_call(
                    logger,
                    f"LLM call completed: {model}",
                    model=model,
                    provider=provider_name,
                    prompt_tokens=(response.usage.get("prompt_tokens") or response.usage.get("input_tokens", 0)) if response.usage else 0,
                    completion_tokens=(response.usage.get("completion_tokens") or response.usage.get("output_tokens", 0)) if response.usage else 0,
                    total_tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                    cost=cost,
                    latency_ms=latency_ms,
                    operation="chat_completion",
                    status=200,
                )
                
                return response.to_dict()
                
            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                last_error = e
                
                if circuit_breaker:
                    circuit_breaker.record_failure()
        
        if hasattr(self, '_mock_model') and self._mock_model in self._providers:
            logger.warning("All configured models failed, falling back to mock provider for demo")
            try:
                mock_provider = self._providers[self._mock_model]
                response = await mock_provider.complete(messages, tools, **merged_params)
                self._track_usage(response.usage, self._mock_model)
                return response.to_dict()
            except Exception as e:
                logger.error(f"Mock provider also failed: {e}")
        
        raise last_error or Exception("All models failed")
    
    async def _complete_with_retry(
        self,
        provider: BaseLLMProvider,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        params: Dict[str, Any],
    ) -> LLMResponse:
        """Complete with retry logic."""
        max_retries = self.routing_config.get("max_retries", 3)
        backoff_ms = self.routing_config.get("retry_backoff_ms", 1000)
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await provider.complete(
                    messages=messages,
                    tools=tools,
                    temperature=params.get("temperature", 0.7),
                    max_tokens=params.get("max_tokens"),
                    **{k: v for k, v in params.items() if k not in ["temperature", "max_tokens"]},
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = (backoff_ms / 1000) * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        raise last_error
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming completion.
        
        Args:
            messages: Chat messages
            tools: Tool definitions
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        merged_params = {**self.parameters, **kwargs}
        
        provider = self._providers.get(self.primary_model)
        if not provider:
            raise ValueError(f"No provider for model: {self.primary_model}")
        
        async for chunk in provider.stream(
            messages=messages,
            tools=tools,
            temperature=merged_params.get("temperature", 0.7),
            max_tokens=merged_params.get("max_tokens"),
        ):
            yield chunk.to_dict()
    
    def _calculate_cost(self, usage: Dict[str, int], model: Optional[str] = None) -> float:
        """Calculate cost for a single LLM call."""
        if not usage:
            return 0.0
        
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        
        model_key = model or self.primary_model
        pricing = self._pricing.get(model_key, self._pricing["default"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _track_usage(self, usage: Dict[str, int], model: Optional[str] = None):
        """Track token usage and cost."""
        if not usage:
            return
        
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        
        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens
        
        cost = self._calculate_cost(usage, model)
        
        self._cost_tracking["input"] += (input_tokens / 1000) * self._pricing.get(model or self.primary_model, self._pricing["default"])["input"]
        self._cost_tracking["output"] += (output_tokens / 1000) * self._pricing.get(model or self.primary_model, self._pricing["default"])["output"]
        self._cost_tracking["total"] += cost

        # Also update process-wide counters surfaced via GET /api/llm-usage
        _GLOBAL_USAGE["prompt_tokens"]     += input_tokens
        _GLOBAL_USAGE["completion_tokens"] += output_tokens
        _GLOBAL_USAGE["total_tokens"]      += (input_tokens + output_tokens)
        _GLOBAL_USAGE["calls"]             += 1
        _GLOBAL_COST["input_usd"]  += (input_tokens  / 1000) * self._pricing.get(model or self.primary_model, self._pricing["default"])["input"]
        _GLOBAL_COST["output_usd"] += (output_tokens / 1000) * self._pricing.get(model or self.primary_model, self._pricing["default"])["output"]
        _GLOBAL_COST["total_usd"]  += cost
        
        logger.info(f"💰 Cost: ${cost:.6f} model={model or self.primary_model} in={input_tokens} out={output_tokens} total_cost=${self._cost_tracking['total']:.6f}")
    
    def get_usage(self) -> Dict[str, int]:
        """Get accumulated token usage."""
        return self._token_usage.copy()
    
    def get_cost(self) -> Dict[str, float]:
        """Get accumulated cost."""
        return self._cost_tracking.copy()
    
    def reset_usage(self):
        """Reset token usage and cost counters."""
        self._token_usage = {"input": 0, "output": 0}
        self._cost_tracking = {"total": 0.0, "input": 0.0, "output": 0.0}
    
    def set_model(self, model: str):
        """Set primary model."""
        self.primary_model = model
        if model not in self._providers:
            provider_name = self._detect_provider(model)
            self._providers[model] = self._create_provider(provider_name, model)
            self._circuit_breakers[model] = CircuitBreakerState()
    
    def set_temperature(self, temperature: float):
        """Set default temperature."""
        self.parameters["temperature"] = temperature
    
    def set_max_tokens(self, max_tokens: int):
        """Set default max tokens."""
        self.parameters["max_tokens"] = max_tokens
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all providers."""
        health = {
            "status": "healthy",
            "primary_model": self.primary_model,
            "providers": {},
        }
        
        for model, provider in self._providers.items():
            try:
                provider_health = await provider.health_check()
                health["providers"][model] = provider_health
            except Exception as e:
                health["providers"][model] = {"status": "unhealthy", "error": str(e)}
                if model == self.primary_model:
                    health["status"] = "degraded"
        
        return health
    
    async def complete_with_vision(
        self,
        messages: List[Dict[str, Any]],
        images: List[Union[str, Dict[str, Any]]],
        detail: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with image inputs.
        
        Args:
            messages: Chat messages
            images: List of image URLs or base64 data
            detail: Image detail level (low, high, auto)
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        if not self.vision_config.get("enabled", False):
            raise ValueError("Vision is not enabled in configuration")
        
        from .providers.openai import OpenAIProvider, ImageContent
        
        provider = self._providers.get(self.primary_model)
        if not isinstance(provider, OpenAIProvider):
            raise ValueError("Vision requires OpenAI provider")
        
        if not provider.supports_vision:
            raise ValueError(f"Model {self.primary_model} does not support vision")
        
        detail = detail or self.vision_config.get("detail", "auto")
        max_images = self.vision_config.get("max_images_per_request", 5)
        
        if len(images) > max_images:
            raise ValueError(f"Too many images: {len(images)} > {max_images}")
        
        merged_params = {**self.parameters, **kwargs}
        
        response = await provider.complete_with_vision(
            messages=messages,
            images=images,
            detail=detail,
            temperature=merged_params.get("temperature", 0.7),
            max_tokens=merged_params.get("max_tokens"),
        )
        
        self._track_usage(response.usage, self.primary_model)
        return response.to_dict()
    
    async def complete_with_audio(
        self,
        messages: List[Dict[str, Any]],
        audio_data: Union[str, bytes],
        audio_format: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with audio input/output.
        
        Args:
            messages: Chat messages
            audio_data: Base64 audio data or bytes
            audio_format: Input audio format
            voice: Output voice
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with audio in metadata
        """
        if not self.audio_config.get("enabled", False):
            raise ValueError("Audio is not enabled in configuration")
        
        from .providers.openai import OpenAIProvider
        
        provider = self._providers.get(self.primary_model)
        if not isinstance(provider, OpenAIProvider):
            raise ValueError("Audio requires OpenAI provider")
        
        if not provider.supports_audio:
            raise ValueError(f"Model {self.primary_model} does not support audio")
        
        audio_format = audio_format or self.audio_config.get("input_formats", ["wav"])[0]
        voice = voice or self.audio_config.get("voice", "alloy")
        output_format = self.audio_config.get("output_format", "wav")
        
        merged_params = {**self.parameters, **kwargs}
        
        response = await provider.complete_with_audio(
            messages=messages,
            audio_data=audio_data,
            audio_format=audio_format,
            voice=voice,
            output_format=output_format,
            temperature=merged_params.get("temperature", 0.7),
            max_tokens=merged_params.get("max_tokens"),
        )
        
        self._track_usage(response.usage, self.primary_model)
        return response.to_dict()
    
    async def complete_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        schema_name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with structured output (JSON Schema).
        
        Args:
            messages: Chat messages
            schema_name: Name of predefined schema
            schema: Custom schema (if not using predefined)
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response
        """
        if not self.structured_output_config.get("enabled", False):
            raise ValueError("Structured output is not enabled in configuration")
        
        from .providers.openai import OpenAIProvider, StructuredOutput
        
        provider = self._providers.get(self.primary_model)
        if not isinstance(provider, OpenAIProvider):
            raise ValueError("Structured output requires OpenAI provider")
        
        if schema_name:
            schemas = self.structured_output_config.get("schemas", {})
            schema_config = schemas.get(schema_name)
            if not schema_config:
                raise ValueError(f"Schema not found: {schema_name}")
            output_schema = StructuredOutput(
                name=schema_config["name"],
                schema=schema_config["schema"],
                strict=schema_config.get("strict", True),
                description=schema_config.get("description"),
            )
        elif schema:
            output_schema = StructuredOutput(
                name=schema.get("name", "custom_schema"),
                schema=schema.get("schema", schema),
                strict=schema.get("strict", self.structured_output_config.get("strict", True)),
            )
        else:
            default_schema = self.structured_output_config.get("default_schema")
            if not default_schema:
                raise ValueError("No schema provided and no default schema configured")
            output_schema = StructuredOutput(
                name="default",
                schema=default_schema,
                strict=self.structured_output_config.get("strict", True),
            )
        
        merged_params = {**self.parameters, **kwargs}
        
        result = await provider.complete_with_structured_output(
            messages=messages,
            output_schema=output_schema,
            temperature=merged_params.get("temperature", 0.7),
            max_tokens=merged_params.get("max_tokens"),
        )
        
        return result
    
    async def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model (uses config default if not specified)
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings_config.get("enabled", False):
            raise ValueError("Embeddings is not enabled in configuration")
        
        from .providers.openai import OpenAIProvider
        
        provider = self._providers.get(self.primary_model)
        if not isinstance(provider, OpenAIProvider):
            raise ValueError("Embeddings requires OpenAI provider")
        
        model = model or self.embeddings_config.get("model", "text-embedding-3-small")
        dimensions = self.embeddings_config.get("dimensions")
        batch_size = self.embeddings_config.get("batch_size", 100)
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await provider.create_embeddings(
                texts=batch,
                model=model,
                dimensions=dimensions,
            )
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def moderate(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Check text for policy violations.
        
        Args:
            text: Text to moderate
            
        Returns:
            Moderation results
        """
        if not self.moderation_config.get("enabled", False):
            raise ValueError("Moderation is not enabled in configuration")
        
        from .providers.openai import OpenAIProvider
        
        provider = self._providers.get(self.primary_model)
        if not isinstance(provider, OpenAIProvider):
            raise ValueError("Moderation requires OpenAI provider")
        
        model = self.moderation_config.get("model", "text-moderation-latest")
        
        result = await provider.moderate(text=text, model=model)
        
        categories_to_check = self.moderation_config.get("categories_to_check")
        threshold_overrides = self.moderation_config.get("threshold_overrides", {})
        
        if categories_to_check:
            filtered_categories = {}
            filtered_scores = {}
            for cat in categories_to_check:
                if cat in result["categories"]:
                    filtered_categories[cat] = result["categories"][cat]
                    filtered_scores[cat] = result["category_scores"][cat]
            result["categories"] = filtered_categories
            result["category_scores"] = filtered_scores
        
        if threshold_overrides:
            for cat, threshold in threshold_overrides.items():
                if cat in result["category_scores"]:
                    if result["category_scores"][cat] >= threshold:
                        result["flagged"] = True
                        result["categories"][cat] = True
        
        return result
    
    async def moderate_input(self, text: str) -> bool:
        """
        Check if input text should be blocked.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text should be blocked
        """
        if not self.moderation_config.get("enabled", False):
            return False
        
        if not self.moderation_config.get("check_input", True):
            return False
        
        result = await self.moderate(text)
        
        if result["flagged"] and self.moderation_config.get("block_on_violation", True):
            logger.warning(f"Input blocked by moderation: {result['categories']}")
            return True
        
        return False
    
    async def moderate_output(self, text: str) -> bool:
        """
        Check if output text should be blocked.
        
        Args:
            text: Output text to check
            
        Returns:
            True if text should be blocked
        """
        if not self.moderation_config.get("enabled", False):
            return False
        
        if not self.moderation_config.get("check_output", False):
            return False
        
        result = await self.moderate(text)
        
        if result["flagged"] and self.moderation_config.get("block_on_violation", True):
            logger.warning(f"Output blocked by moderation: {result['categories']}")
            return True
        
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get router capabilities based on configuration."""
        from .providers.openai import OpenAIProvider
        
        provider = self._providers.get(self.primary_model)
        is_openai = isinstance(provider, OpenAIProvider)
        
        return {
            "vision": is_openai and self.vision_config.get("enabled", False) and provider.supports_vision if is_openai else False,
            "audio": is_openai and self.audio_config.get("enabled", False) and provider.supports_audio if is_openai else False,
            "structured_output": is_openai and self.structured_output_config.get("enabled", False) and provider.supports_structured_output if is_openai else False,
            "embeddings": self.embeddings_config.get("enabled", False),
            "moderation": self.moderation_config.get("enabled", False),
            "streaming": self.streaming_config.get("enabled", True),
            "tool_calling": True,
            "azure_openai": bool(self.azure_config),
        }
    
    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
