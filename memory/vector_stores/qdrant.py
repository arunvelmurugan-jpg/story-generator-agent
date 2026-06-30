"""
Qdrant Vector Store
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig(VectorStoreConfig):
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    grpc_port: int = 6334
    prefer_grpc: bool = False


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: QdrantConfig):
        super().__init__(config)
        self.config: QdrantConfig = config
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.config.url, api_key=self.config.api_key, prefer_grpc=self.config.prefer_grpc)
            except ImportError:
                raise ImportError("qdrant-client required: pip install qdrant-client")
        return self._client
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        from qdrant_client.models import PointStruct, VectorParams, Distance
        client = self._get_client()
        
        try:
            client.get_collection(self.config.collection_name)
        except Exception:
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=self.config.embedding_dimension, distance=Distance.COSINE)
            )
        
        points = [PointStruct(id=i, vector=doc.embedding, payload={"content": doc.content, "doc_id": doc.id, **doc.metadata})
                  for i, doc in enumerate(documents) if doc.embedding]
        
        if points:
            client.upsert(collection_name=self.config.collection_name, points=points)
        
        return [doc.id for doc in documents]
    
    async def search(self, query_embedding: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        client = self._get_client()
        results = client.search(collection_name=self.config.collection_name, query_vector=query_embedding, limit=top_k)
        
        return [SearchResult(
            id=str(r.payload.get("doc_id", r.id)),
            content=r.payload.get("content", ""),
            score=r.score,
            metadata={k: v for k, v in r.payload.items() if k not in ["content", "doc_id"]}
        ) for r in results]
    
    async def delete(self, ids: List[str]) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        client = self._get_client()
        client.delete(collection_name=self.config.collection_name, points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=ids))]))
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        client = self._get_client()
        results = client.scroll(collection_name=self.config.collection_name, limit=len(ids))[0]
        return [Document(id=str(r.payload.get("doc_id", r.id)), content=r.payload.get("content", ""), metadata=r.payload) for r in results if str(r.payload.get("doc_id", r.id)) in ids]


__all__ = ["QdrantVectorStore", "QdrantConfig"]
