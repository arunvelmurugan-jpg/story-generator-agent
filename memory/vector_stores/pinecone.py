"""
Pinecone Vector Store
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import os

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig(VectorStoreConfig):
    api_key: Optional[str] = None
    environment: str = "us-east-1"
    index_name: str = "default"
    namespace: str = ""


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: PineconeConfig):
        super().__init__(config)
        self.config: PineconeConfig = config
        if not self.config.api_key:
            self.config.api_key = os.environ.get("PINECONE_API_KEY")
        self._index = None
    
    def _get_index(self):
        if self._index is None:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.config.api_key)
                self._index = pc.Index(self.config.index_name)
            except ImportError:
                raise ImportError("pinecone-client required: pip install pinecone-client")
        return self._index
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        index = self._get_index()
        vectors = [(doc.id, doc.embedding, {"content": doc.content, **doc.metadata}) for doc in documents if doc.embedding]
        if vectors:
            index.upsert(vectors=vectors, namespace=self.config.namespace)
        return [doc.id for doc in documents]
    
    async def search(self, query_embedding: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        index = self._get_index()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=self.config.namespace, filter=filter)
        return [SearchResult(id=m.id, content=m.metadata.get("content", ""), score=m.score, metadata={k: v for k, v in m.metadata.items() if k != "content"}) for m in results.matches]
    
    async def delete(self, ids: List[str]) -> bool:
        index = self._get_index()
        index.delete(ids=ids, namespace=self.config.namespace)
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        index = self._get_index()
        results = index.fetch(ids=ids, namespace=self.config.namespace)
        return [Document(id=id, content=v.metadata.get("content", ""), embedding=v.values, metadata=v.metadata) for id, v in results.vectors.items()]


__all__ = ["PineconeVectorStore", "PineconeConfig"]
