"""
Memory Manager for PHTN.AI Sub-Agent Framework

Manages different types of memory for agent context.
Now includes full vector store integration for semantic memory.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from .vector_store import (
    VectorStoreConfig,
    VectorStoreProvider,
    VectorDocument,
    BaseVectorStore,
    create_vector_store,
)
from .embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    BaseEmbeddingProvider,
    create_embedding_provider,
)
from ..observability.otel_logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Manages agent memory across multiple types.
    
    Memory Types:
    - Short-term: Session-scoped, in-memory
    - Long-term: Persistent storage (Redis, PostgreSQL)
    - Semantic: Vector-based similarity search (Pinecone, Milvus, etc.)
    - Episodic: Conversation history
    """
    
    def __init__(
        self,
        short_term_config: Optional[Dict[str, Any]] = None,
        long_term_config: Optional[Dict[str, Any]] = None,
        semantic_config: Optional[Dict[str, Any]] = None,
        episodic_config: Optional[Dict[str, Any]] = None,
        token_budget: Optional[int] = None,
    ):
        """
        Initialize MemoryManager.
        
        Args:
            short_term_config: Short-term memory configuration
            long_term_config: Long-term memory configuration
            semantic_config: Semantic memory configuration
            episodic_config: Episodic memory configuration
            token_budget: Token budget for context
        """
        self.short_term_config = short_term_config or {"enabled": True}
        self.long_term_config = long_term_config or {"enabled": False}
        self.semantic_config = semantic_config or {"enabled": False}
        self.episodic_config = episodic_config or {"enabled": True}
        self.token_budget = token_budget
        
        self._short_term: Dict[str, OrderedDict] = {}
        self._episodic: Dict[str, List[Dict[str, Any]]] = {}
        self._long_term_store: Optional[Any] = None
        
        self._vector_store: Optional[BaseVectorStore] = None
        self._embedding_provider: Optional[BaseEmbeddingProvider] = None
        
        self._max_short_term = self.short_term_config.get("max_entries", 100)
        self._max_episodic = self.episodic_config.get("max_episodes", 50)
        
        self._initialized = False
        logger.debug("MemoryManager initialized")
    
    async def initialize(self):
        """Initialize memory backends."""
        if self._initialized:
            return
        
        if self.semantic_config.get("enabled", False):
            await self._init_semantic_memory()
        
        if self.long_term_config.get("enabled", False):
            await self._init_long_term_memory()
        
        self._initialized = True
        logger.info("MemoryManager backends initialized")
    
    async def _init_semantic_memory(self):
        """Initialize semantic memory with vector store."""
        try:
            provider_str = self.semantic_config.get("provider", "in_memory")
            try:
                provider = VectorStoreProvider(provider_str.lower())
            except ValueError:
                provider = VectorStoreProvider.IN_MEMORY
            
            vector_config = VectorStoreConfig(
                provider=provider,
                index_name=self.semantic_config.get("index_name", "agent-memory"),
                dimension=self.semantic_config.get("dimension", 1536),
                metric=self.semantic_config.get("metric", "cosine"),
                api_key=self.semantic_config.get("api_key"),
                environment=self.semantic_config.get("environment"),
                host=self.semantic_config.get("host"),
                port=self.semantic_config.get("port"),
                namespace=self.semantic_config.get("namespace"),
                collection_name=self.semantic_config.get("collection_name"),
            )
            
            self._vector_store = create_vector_store(vector_config)
            await self._vector_store.initialize()
            
            embedding_model = self.semantic_config.get("embedding_model", "text-embedding-3-small")
            embedding_provider_str = self.semantic_config.get("embedding_provider", "openai")
            
            try:
                embedding_provider = EmbeddingProvider(embedding_provider_str.lower())
            except ValueError:
                embedding_provider = EmbeddingProvider.LOCAL
            
            embedding_config = EmbeddingConfig(
                provider=embedding_provider,
                model=embedding_model,
                dimension=self.semantic_config.get("dimension", 1536),
            )
            
            self._embedding_provider = create_embedding_provider(embedding_config)
            
            logger.info(f"Semantic memory initialized with {provider} vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {e}")
            self._vector_store = None
            self._embedding_provider = None
    
    async def _init_long_term_memory(self):
        """Initialize long-term memory storage."""
        provider = self.long_term_config.get("provider", "redis")
        
        if provider == "redis":
            await self._init_redis_storage()
        elif provider == "postgresql":
            await self._init_postgres_storage()
        else:
            logger.warning(f"Unknown long-term memory provider: {provider}")
    
    async def _init_redis_storage(self):
        """Initialize Redis for long-term memory."""
        try:
            import redis.asyncio as redis
            
            host = self.long_term_config.get("host", "localhost")
            port = self.long_term_config.get("port", 6379)
            db = self.long_term_config.get("db", 0)
            password = self.long_term_config.get("password")
            
            self._long_term_store = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            
            await self._long_term_store.ping()
            logger.info("Redis long-term memory initialized")
            
        except ImportError:
            logger.error("Redis package not installed. Run: pip install redis")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
    
    async def _init_postgres_storage(self):
        """Initialize PostgreSQL for long-term memory."""
        logger.warning("PostgreSQL long-term memory not yet implemented")
    
    async def store_input(
        self,
        session_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store input in memory."""
        await self._add_to_episodic(session_id, "user", content, metadata)
        await self._add_to_short_term(session_id, "input", content, metadata)
    
    async def store_output(
        self,
        session_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store output in memory."""
        await self._add_to_episodic(session_id, "assistant", content, metadata)
        await self._add_to_short_term(session_id, "output", content, metadata)
    
    async def get_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Get context from memory for a session.
        
        Args:
            session_id: Session identifier
            max_tokens: Maximum tokens to include
            
        Returns:
            Context string or None
        """
        if not self.episodic_config.get("enabled", True):
            return None
        
        episodes = self._episodic.get(session_id, [])
        
        if not episodes:
            return None
        
        context_parts = []
        for episode in episodes[-10:]:
            role = episode.get("role", "unknown")
            content = episode.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memory.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of matching documents
        """
        if not self.semantic_config.get("enabled", False):
            logger.warning("Semantic memory not enabled")
            return []
        
        if not self._vector_store or not self._embedding_provider:
            logger.warning("Semantic memory not initialized")
            return []
        
        try:
            query_embedding = await self._embedding_provider.embed_single(query)
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            documents = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            
            return [doc.to_dict() for doc in documents]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def add_to_semantic(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Add content to semantic memory.
        
        Args:
            content: Content to store
            metadata: Optional metadata
            doc_id: Optional document ID
        """
        if not self.semantic_config.get("enabled", False):
            return
        
        if not self._vector_store or not self._embedding_provider:
            logger.warning("Semantic memory not initialized")
            return
        
        try:
            if not doc_id:
                doc_id = hashlib.md5(content.encode()).hexdigest()
            
            embedding = await self._embedding_provider.embed_single(content)
            
            if not embedding:
                logger.warning("Failed to generate embedding")
                return
            
            document = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.utcnow().isoformat(),
            )
            
            await self._vector_store.upsert([document])
            logger.debug(f"Added document to semantic memory: {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to add to semantic memory: {e}")
    
    async def add_documents_to_semantic(
        self,
        documents: List[Dict[str, Any]],
    ):
        """
        Add multiple documents to semantic memory.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata', 'id'
        """
        if not self.semantic_config.get("enabled", False):
            return
        
        if not self._vector_store or not self._embedding_provider:
            logger.warning("Semantic memory not initialized")
            return
        
        try:
            contents = [doc.get("content", "") for doc in documents]
            embeddings = await self._embedding_provider.embed(contents)
            
            vector_docs = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id") or hashlib.md5(contents[i].encode()).hexdigest()
                vector_docs.append(VectorDocument(
                    id=doc_id,
                    content=contents[i],
                    embedding=embeddings[i] if i < len(embeddings) else [],
                    metadata=doc.get("metadata", {}),
                    created_at=datetime.utcnow().isoformat(),
                ))
            
            await self._vector_store.upsert(vector_docs)
            logger.info(f"Added {len(vector_docs)} documents to semantic memory")
            
        except Exception as e:
            logger.error(f"Failed to add documents to semantic memory: {e}")
    
    async def delete_from_semantic(self, doc_ids: List[str]):
        """Delete documents from semantic memory."""
        if self._vector_store:
            await self._vector_store.delete(doc_ids)
    
    async def store_long_term(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Store data in long-term memory.
        
        Args:
            key: Storage key
            value: Value to store
            ttl_seconds: Optional TTL
        """
        if not self.long_term_config.get("enabled", False):
            return
        
        if not self._long_term_store:
            logger.warning("Long-term memory not initialized")
            return
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl_seconds:
                await self._long_term_store.setex(key, ttl_seconds, value)
            else:
                await self._long_term_store.set(key, value)
                
        except Exception as e:
            logger.error(f"Failed to store in long-term memory: {e}")
    
    async def get_long_term(self, key: str) -> Optional[Any]:
        """
        Get data from long-term memory.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None
        """
        if not self.long_term_config.get("enabled", False):
            return None
        
        if not self._long_term_store:
            return None
        
        try:
            value = await self._long_term_store.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get from long-term memory: {e}")
            return None
    
    async def delete_long_term(self, key: str):
        """Delete data from long-term memory."""
        if self._long_term_store:
            try:
                await self._long_term_store.delete(key)
            except Exception as e:
                logger.error(f"Failed to delete from long-term memory: {e}")
    
    async def _add_to_short_term(
        self,
        session_id: str,
        entry_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add entry to short-term memory."""
        if not self.short_term_config.get("enabled", True):
            return
        
        if session_id not in self._short_term:
            self._short_term[session_id] = OrderedDict()
        
        session_memory = self._short_term[session_id]
        
        entry_id = f"{entry_type}_{datetime.utcnow().timestamp()}"
        session_memory[entry_id] = {
            "type": entry_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        while len(session_memory) > self._max_short_term:
            session_memory.popitem(last=False)
    
    async def _add_to_episodic(
        self,
        session_id: str,
        role: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add entry to episodic memory."""
        if not self.episodic_config.get("enabled", True):
            return
        
        if session_id not in self._episodic:
            self._episodic[session_id] = []
        
        self._episodic[session_id].append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        while len(self._episodic[session_id]) > self._max_episodic:
            self._episodic[session_id].pop(0)
    
    async def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        self._short_term.pop(session_id, None)
        self._episodic.pop(session_id, None)
    
    async def close(self):
        """Close memory manager and cleanup."""
        self._short_term.clear()
        self._episodic.clear()
        
        if self._vector_store:
            await self._vector_store.close()
            self._vector_store = None
        
        if self._long_term_store:
            try:
                await self._long_term_store.close()
            except Exception:
                pass
            self._long_term_store = None
        
        self._initialized = False
        logger.info("MemoryManager closed")
