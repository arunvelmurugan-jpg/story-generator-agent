"""
In-Memory Vector Store
"""

import logging
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development and testing."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        super().__init__(config or VectorStoreConfig())
        self._documents: Dict[str, Document] = {}
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        for doc in documents:
            self._documents[doc.id] = doc
        return [doc.id for doc in documents]
    
    async def search(self, query_embedding: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        results = []
        for doc in self._documents.values():
            if not doc.embedding:
                continue
            
            if filter:
                match = all(doc.metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            
            score = self._cosine_similarity(query_embedding, doc.embedding)
            results.append(SearchResult(id=doc.id, content=doc.content, score=score, metadata=doc.metadata))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def delete(self, ids: List[str]) -> bool:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        return [self._documents[doc_id] for doc_id in ids if doc_id in self._documents]
    
    async def clear(self) -> bool:
        self._documents.clear()
        return True


__all__ = ["InMemoryVectorStore"]
