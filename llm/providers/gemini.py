"""
Google Gemini/Vertex AI LLM Provider

Supports Google's AI models:
- Gemini Pro
- Gemini Pro Vision
- Gemini Ultra
- Gemini 1.5 Pro
- Gemini 1.5 Flash
- PaLM 2 (via Vertex AI)

Equivalent to n8n's Google Gemini Chat Model node.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass

from ..base import BaseLLMProvider, LLMResponse, StreamChunk, Message

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini provider."""
    api_key: Optional[str] = None
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: Optional[List[str]] = None
    safety_settings: Optional[Dict[str, str]] = None
    project_id: Optional[str] = None
    location: str = "us-central1"
    use_vertex_ai: bool = False


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider.
    
    Supports both Google AI Studio (API key) and Vertex AI (service account):
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-1.0-pro
    - gemini-1.0-pro-vision
    - gemini-ultra (limited access)
    """
    
    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision",
        "gemini-pro",
        "gemini-pro-vision",
        "gemini-ultra",
        "text-bison@002",
        "chat-bison@002",
        "codechat-bison@002",
    ]
    
    def __init__(self, config: Optional[GeminiConfig] = None, **kwargs):
        self.config = config or GeminiConfig(**kwargs)
        self._client = None
        self._model = None
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
    
    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            if self.config.use_vertex_ai:
                self._init_vertex_ai()
            else:
                self._init_google_ai()
        return self._client
    
    def _init_google_ai(self):
        """Initialize Google AI Studio client."""
        try:
            import google.generativeai as genai
            
            if not self.config.api_key:
                import os
                self.config.api_key = os.environ.get("GOOGLE_API_KEY")
            
            if not self.config.api_key:
                raise ValueError("GOOGLE_API_KEY is required for Google AI Studio")
            
            genai.configure(api_key=self.config.api_key)
            
            generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            }
            
            if self.config.stop_sequences:
                generation_config["stop_sequences"] = self.config.stop_sequences
            
            safety_settings = self.config.safety_settings or {}
            
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            self._client = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini provider. "
                "Install with: pip install google-generativeai"
            )
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            if not self.config.project_id:
                import os
                self.config.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location
            )
            
            self._model = GenerativeModel(self.config.model)
            self._client = vertexai
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI. "
                "Install with: pip install google-cloud-aiplatform"
            )
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API."""
        formatted = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                formatted.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                formatted.append({"role": "model", "parts": [msg.content]})
        
        return formatted, system_instruction
    
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        self._get_client()
        
        if model and model != self.config.model:
            if self.config.use_vertex_ai:
                from vertexai.generative_models import GenerativeModel
                self._model = GenerativeModel(model)
            else:
                import google.generativeai as genai
                self._model = genai.GenerativeModel(model)
        
        formatted_messages, system_instruction = self._format_messages(messages)
        
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            if system_instruction:
                chat = self._model.start_chat(history=formatted_messages[:-1])
                user_message = formatted_messages[-1]["parts"][0] if formatted_messages else ""
                
                if self.config.use_vertex_ai:
                    response = chat.send_message(
                        f"{system_instruction}\n\n{user_message}",
                        generation_config=generation_config if generation_config else None
                    )
                else:
                    response = chat.send_message(
                        f"{system_instruction}\n\n{user_message}",
                        generation_config=generation_config if generation_config else None
                    )
            else:
                chat = self._model.start_chat(history=formatted_messages[:-1])
                user_message = formatted_messages[-1]["parts"][0] if formatted_messages else ""
                response = chat.send_message(
                    user_message,
                    generation_config=generation_config if generation_config else None
                )
            
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
                }
            
            return LLMResponse(
                content=response.text,
                model=model or self.config.model,
                provider=self.provider_name,
                usage=usage,
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                    "safety_ratings": [
                        {"category": r.category.name, "probability": r.probability.name}
                        for r in response.candidates[0].safety_ratings
                    ] if response.candidates else []
                }
            )
        except Exception as e:
            logger.error(f"Gemini completion error: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion using Gemini."""
        self._get_client()
        
        if model and model != self.config.model:
            if self.config.use_vertex_ai:
                from vertexai.generative_models import GenerativeModel
                self._model = GenerativeModel(model)
            else:
                import google.generativeai as genai
                self._model = genai.GenerativeModel(model)
        
        formatted_messages, system_instruction = self._format_messages(messages)
        
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            chat = self._model.start_chat(history=formatted_messages[:-1])
            user_message = formatted_messages[-1]["parts"][0] if formatted_messages else ""
            
            if system_instruction:
                user_message = f"{system_instruction}\n\n{user_message}"
            
            response = chat.send_message(
                user_message,
                generation_config=generation_config if generation_config else None,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield StreamChunk(content=chunk.text, done=False)
            
            yield StreamChunk(content="", done=True)
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using Gemini embedding model."""
        self._get_client()
        
        embedding_model = model or "models/embedding-001"
        
        try:
            if self.config.use_vertex_ai:
                from vertexai.language_models import TextEmbeddingModel
                model_instance = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
                embeddings = model_instance.get_embeddings(texts)
                return [e.values for e in embeddings]
            else:
                import google.generativeai as genai
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model=embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result["embedding"])
                return embeddings
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            raise
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[bytes],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with image inputs."""
        self._get_client()
        
        vision_model = model or "gemini-1.5-pro"
        
        try:
            if self.config.use_vertex_ai:
                from vertexai.generative_models import GenerativeModel, Part, Image
                model_instance = GenerativeModel(vision_model)
                
                parts = [prompt]
                for img_bytes in images:
                    parts.append(Part.from_image(Image.from_bytes(img_bytes)))
                
                response = model_instance.generate_content(parts)
            else:
                import google.generativeai as genai
                from PIL import Image
                import io
                
                model_instance = genai.GenerativeModel(vision_model)
                
                parts = [prompt]
                for img_bytes in images:
                    img = Image.open(io.BytesIO(img_bytes))
                    parts.append(img)
                
                response = model_instance.generate_content(parts)
            
            return LLMResponse(
                content=response.text,
                model=vision_model,
                provider=self.provider_name,
                usage={},
                metadata={"vision": True}
            )
        except Exception as e:
            logger.error(f"Gemini vision error: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        self._get_client()
        
        try:
            result = self._model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.error(f"Gemini token count error: {e}")
            return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check Gemini API connectivity."""
        try:
            self._get_client()
            return self._model is not None
        except Exception:
            return False


__all__ = ["GeminiProvider", "GeminiConfig"]
