"""
Embeddings Provider for PHTN.AI Sub-Agent Framework

Provides text embedding generation using various providers.
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

from ..observability.otel_logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    api_key: Optional[str] = None
    batch_size: int = 100
    timeout_ms: int = 30000


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._cache: Dict[str, List[float]] = {}
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embeddings = await self.embed([text])
        if embeddings:
            self._cache[cache_key] = embeddings[0]
            return embeddings[0]
        return []
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            import openai
            import os
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not configured")
                return [[] for _ in texts]
            
            client = openai.AsyncOpenAI(api_key=api_key)
            
            all_embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                response = await client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return [[] for _ in texts]


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider."""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            import cohere
            import os
            
            api_key = self.config.api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                logger.error("Cohere API key not configured")
                return [[] for _ in texts]
            
            client = cohere.Client(api_key)
            
            response = client.embed(
                texts=texts,
                model=self.config.model or "embed-english-v3.0",
                input_type="search_document",
            )
            
            return response.embeddings
            
        except ImportError:
            logger.error("Cohere package not installed. Run: pip install cohere")
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            return [[] for _ in texts]


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider (local or API)."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._model is None:
                model_name = self.config.model or "all-MiniLM-L6-v2"
                self._model = SentenceTransformer(model_name)
                logger.info(f"Loaded HuggingFace model: {model_name}")
            
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
            
            return embeddings.tolist()
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            return [[] for _ in texts]


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using simple hashing (for testing)."""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes]
            while len(embedding) < self.config.dimension:
                embedding.extend(embedding)
            embedding = embedding[:self.config.dimension]
            embeddings.append(embedding)
        return embeddings


def create_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """
    Factory function to create embedding provider.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Embedding provider instance
    """
    providers = {
        EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProvider.COHERE: CohereEmbeddingProvider,
        EmbeddingProvider.HUGGINGFACE: HuggingFaceEmbeddingProvider,
        EmbeddingProvider.LOCAL: LocalEmbeddingProvider,
    }
    
    provider_class = providers.get(config.provider, LocalEmbeddingProvider)
    return provider_class(config)
