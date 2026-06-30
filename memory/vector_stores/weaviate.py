"""
Weaviate Vector Store
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class WeaviateConfig(VectorStoreConfig):
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    class_name: str = "Document"


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(self, config: WeaviateConfig):
        super().__init__(config)
        self.config: WeaviateConfig = config
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import weaviate
                auth = weaviate.auth.AuthApiKey(self.config.api_key) if self.config.api_key else None
                self._client = weaviate.Client(url=self.config.url, auth_client_secret=auth)
            except ImportError:
                raise ImportError("weaviate-client required: pip install weaviate-client")
        return self._client
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        client = self._get_client()
        ids = []
        for doc in documents:
            result = client.data_object.create(
                data_object={"content": doc.content, **doc.metadata},
                class_name=self.config.class_name,
                uuid=doc.id,
                vector=doc.embedding
            )
            ids.append(doc.id)
        return ids
    
    async def search(self, query_embedding: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        client = self._get_client()
        query = client.query.get(self.config.class_name, ["content"]).with_near_vector({"vector": query_embedding}).with_limit(top_k).with_additional(["id", "certainty"])
        result = query.do()
        
        results = []
        data = result.get("data", {}).get("Get", {}).get(self.config.class_name, [])
        for item in data:
            results.append(SearchResult(
                id=item.get("_additional", {}).get("id", ""),
                content=item.get("content", ""),
                score=item.get("_additional", {}).get("certainty", 0),
                metadata={k: v for k, v in item.items() if k not in ["content", "_additional"]}
            ))
        return results
    
    async def delete(self, ids: List[str]) -> bool:
        client = self._get_client()
        for doc_id in ids:
            client.data_object.delete(uuid=doc_id, class_name=self.config.class_name)
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        client = self._get_client()
        documents = []
        for doc_id in ids:
            obj = client.data_object.get_by_id(doc_id, class_name=self.config.class_name)
            if obj:
                documents.append(Document(id=doc_id, content=obj.get("properties", {}).get("content", ""), metadata=obj.get("properties", {})))
        return documents


__all__ = ["WeaviateVectorStore", "WeaviateConfig"]
