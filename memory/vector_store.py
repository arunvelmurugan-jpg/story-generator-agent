"""
Vector Store for PHTN.AI Sub-Agent Framework

Provides vector database integration for semantic memory.
Supports multiple providers: Pinecone, Milvus, Weaviate, Qdrant, ChromaDB.
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from ..observability.otel_logging import get_logger

logger = get_logger(__name__)


class VectorStoreProvider(str, Enum):
    """Supported vector store providers."""
    PINECONE = "pinecone"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    IN_MEMORY = "in_memory"


@dataclass
class VectorDocument:
    """Document stored in vector database."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "created_at": self.created_at,
        }


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    provider: VectorStoreProvider = VectorStoreProvider.IN_MEMORY
    index_name: str = "default"
    dimension: int = 1536
    metric: str = "cosine"
    
    api_key: Optional[str] = None
    environment: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    
    namespace: Optional[str] = None
    collection_name: Optional[str] = None
    
    batch_size: int = 100
    timeout_ms: int = 30000


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector store connection."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        documents: List[VectorDocument],
    ) -> bool:
        """Insert or update documents."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: List[str],
    ) -> bool:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the connection."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development/testing."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._documents: Dict[str, VectorDocument] = {}
    
    async def initialize(self):
        self._initialized = True
        logger.info("In-memory vector store initialized")
    
    async def upsert(self, documents: List[VectorDocument]) -> bool:
        for doc in documents:
            self._documents[doc.id] = doc
        return True
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        results = []
        for doc in self._documents.values():
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                
                if filters:
                    if not self._matches_filters(doc.metadata, filters):
                        continue
                
                doc_copy = VectorDocument(
                    id=doc.id,
                    content=doc.content,
                    embedding=doc.embedding,
                    metadata=doc.metadata,
                    score=score,
                    created_at=doc.created_at,
                )
                results.append(doc_copy)
        
        results.sort(key=lambda x: x.score or 0, reverse=True)
        return results[:top_k]
    
    async def delete(self, ids: List[str]) -> bool:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        return True
    
    async def close(self):
        self._documents.clear()
        self._initialized = False
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._index = None
    
    async def initialize(self):
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            pc = Pinecone(api_key=self.config.api_key)
            
            existing_indexes = pc.list_indexes().names()
            if self.config.index_name not in existing_indexes:
                pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.config.environment or "us-east-1",
                    ),
                )
                logger.info(f"Created Pinecone index: {self.config.index_name}")
            
            self._index = pc.Index(self.config.index_name)
            self._initialized = True
            logger.info(f"Connected to Pinecone index: {self.config.index_name}")
            
        except ImportError:
            logger.error("Pinecone package not installed. Run: pip install pinecone-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def upsert(self, documents: List[VectorDocument]) -> bool:
        if not self._index:
            return False
        
        try:
            vectors = []
            for doc in documents:
                if doc.embedding:
                    vectors.append({
                        "id": doc.id,
                        "values": doc.embedding,
                        "metadata": {
                            "content": doc.content[:1000],
                            **doc.metadata,
                        },
                    })
            
            if vectors:
                for i in range(0, len(vectors), self.config.batch_size):
                    batch = vectors[i:i + self.config.batch_size]
                    self._index.upsert(
                        vectors=batch,
                        namespace=self.config.namespace,
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        if not self._index:
            return []
        
        try:
            results = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.config.namespace,
                filter=filters,
            )
            
            documents = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                content = metadata.pop("content", "")
                documents.append(VectorDocument(
                    id=match["id"],
                    content=content,
                    score=match.get("score"),
                    metadata=metadata,
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    async def delete(self, ids: List[str]) -> bool:
        if not self._index:
            return False
        
        try:
            self._index.delete(
                ids=ids,
                namespace=self.config.namespace,
            )
            return True
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")
            return False
    
    async def close(self):
        self._index = None
        self._initialized = False


class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._client = None
    
    async def initialize(self):
        try:
            from pymilvus import MilvusClient
            
            uri = f"http://{self.config.host or 'localhost'}:{self.config.port or 19530}"
            self._client = MilvusClient(uri=uri)
            
            collection_name = self.config.collection_name or self.config.index_name
            
            if not self._client.has_collection(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=self.config.dimension,
                    metric_type="COSINE" if self.config.metric == "cosine" else "L2",
                )
                logger.info(f"Created Milvus collection: {collection_name}")
            
            self._initialized = True
            logger.info(f"Connected to Milvus collection: {collection_name}")
            
        except ImportError:
            logger.error("PyMilvus package not installed. Run: pip install pymilvus")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise
    
    async def upsert(self, documents: List[VectorDocument]) -> bool:
        if not self._client:
            return False
        
        try:
            collection_name = self.config.collection_name or self.config.index_name
            
            data = []
            for doc in documents:
                if doc.embedding:
                    data.append({
                        "id": doc.id,
                        "vector": doc.embedding,
                        "content": doc.content,
                        **doc.metadata,
                    })
            
            if data:
                self._client.upsert(
                    collection_name=collection_name,
                    data=data,
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Milvus upsert failed: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        if not self._client:
            return []
        
        try:
            collection_name = self.config.collection_name or self.config.index_name
            
            filter_expr = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f'{key} == "{value}"')
                    else:
                        conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions) if conditions else None
            
            results = self._client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=["content"],
            )
            
            documents = []
            for hits in results:
                for hit in hits:
                    documents.append(VectorDocument(
                        id=str(hit.get("id")),
                        content=hit.get("entity", {}).get("content", ""),
                        score=hit.get("distance"),
                        metadata=hit.get("entity", {}),
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []
    
    async def delete(self, ids: List[str]) -> bool:
        if not self._client:
            return False
        
        try:
            collection_name = self.config.collection_name or self.config.index_name
            self._client.delete(
                collection_name=collection_name,
                ids=ids,
            )
            return True
        except Exception as e:
            logger.error(f"Milvus delete failed: {e}")
            return False
    
    async def close(self):
        if self._client:
            self._client.close()
        self._client = None
        self._initialized = False


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._client = None
    
    async def initialize(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            if self.config.api_key:
                self._client = QdrantClient(
                    url=self.config.host,
                    api_key=self.config.api_key,
                )
            else:
                self._client = QdrantClient(
                    host=self.config.host or "localhost",
                    port=self.config.port or 6333,
                )
            
            collection_name = self.config.collection_name or self.config.index_name
            
            collections = self._client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                distance = Distance.COSINE if self.config.metric == "cosine" else Distance.EUCLID
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=distance,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            self._initialized = True
            logger.info(f"Connected to Qdrant collection: {collection_name}")
            
        except ImportError:
            logger.error("Qdrant client not installed. Run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def upsert(self, documents: List[VectorDocument]) -> bool:
        if not self._client:
            return False
        
        try:
            from qdrant_client.models import PointStruct
            
            collection_name = self.config.collection_name or self.config.index_name
            
            points = []
            for doc in documents:
                if doc.embedding:
                    points.append(PointStruct(
                        id=hash(doc.id) % (2**63),
                        vector=doc.embedding,
                        payload={
                            "doc_id": doc.id,
                            "content": doc.content,
                            **doc.metadata,
                        },
                    ))
            
            if points:
                self._client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorDocument]:
        if not self._client:
            return []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            collection_name = self.config.collection_name or self.config.index_name
            
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    ))
                query_filter = Filter(must=conditions) if conditions else None
            
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
            )
            
            documents = []
            for hit in results:
                payload = hit.payload or {}
                documents.append(VectorDocument(
                    id=payload.get("doc_id", str(hit.id)),
                    content=payload.get("content", ""),
                    score=hit.score,
                    metadata={k: v for k, v in payload.items() if k not in ["doc_id", "content"]},
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    async def delete(self, ids: List[str]) -> bool:
        if not self._client:
            return False
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            
            collection_name = self.config.collection_name or self.config.index_name
            self._client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchAny(any=ids))]
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            return False
    
    async def close(self):
        if self._client:
            self._client.close()
        self._client = None
        self._initialized = False


def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
    """
    Factory function to create vector store instance.
    
    Args:
        config: Vector store configuration
        
    Returns:
        Vector store instance
    """
    providers = {
        VectorStoreProvider.PINECONE: PineconeVectorStore,
        VectorStoreProvider.MILVUS: MilvusVectorStore,
        VectorStoreProvider.QDRANT: QdrantVectorStore,
        VectorStoreProvider.IN_MEMORY: InMemoryVectorStore,
    }
    
    store_class = providers.get(config.provider, InMemoryVectorStore)
    return store_class(config)
