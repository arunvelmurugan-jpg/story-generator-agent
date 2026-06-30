"""
Base Vector Store Interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Document to store in vector database."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result from vector database."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Base configuration for vector stores."""
    collection_name: str = "default"
    embedding_dimension: int = 1536
    metric: str = "cosine"
    batch_size: int = 100


class BaseVectorStore(ABC):
    """Base class for all vector stores."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    async def get(self, ids: List[str]) -> List[Document]:
        """Get documents by ID."""
        pass
    
    async def upsert(self, documents: List[Document]) -> List[str]:
        """Upsert documents (add or update)."""
        return await self.add_documents(documents)
    
    async def clear(self) -> bool:
        """Clear all documents from the collection."""
        return True


__all__ = ["BaseVectorStore", "VectorStoreConfig", "Document", "SearchResult"]
