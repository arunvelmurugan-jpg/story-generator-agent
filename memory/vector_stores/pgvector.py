"""
PGVector Store

PostgreSQL with pgvector extension.
Equivalent to n8n's Postgres Vector Store node.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class PGVectorConfig(VectorStoreConfig):
    """PGVector-specific configuration."""
    connection_string: str = ""
    host: str = "localhost"
    port: int = 5432
    database: str = "vectors"
    user: str = "postgres"
    password: str = ""
    table_name: str = "embeddings"
    create_extension: bool = True


class PGVectorStore(BaseVectorStore):
    """
    PostgreSQL vector store with pgvector extension.
    
    Features:
    - SQL-based queries
    - ACID compliance
    - Metadata filtering
    - Hybrid search
    """
    
    def __init__(self, config: PGVectorConfig):
        super().__init__(config)
        self.config: PGVectorConfig = config
        self._pool = None
    
    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                
                if self.config.connection_string:
                    self._pool = await asyncpg.create_pool(self.config.connection_string)
                else:
                    self._pool = await asyncpg.create_pool(
                        host=self.config.host,
                        port=self.config.port,
                        database=self.config.database,
                        user=self.config.user,
                        password=self.config.password
                    )
                
                if self.config.create_extension:
                    await self._setup_database()
                    
            except ImportError:
                raise ImportError("asyncpg required: pip install asyncpg")
        return self._pool
    
    async def _setup_database(self):
        """Set up pgvector extension and table."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding vector({self.config.embedding_dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.config.table_name}_embedding_idx
                ON {self.config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to PGVector."""
        pool = await self._get_pool()
        ids = []
        
        async with pool.acquire() as conn:
            for doc in documents:
                import json
                await conn.execute(f"""
                    INSERT INTO {self.config.table_name} (id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, doc.id, doc.content, doc.embedding, json.dumps(doc.metadata))
                ids.append(doc.id)
        
        return ids
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search PGVector for similar documents."""
        pool = await self._get_pool()
        
        where_clause = ""
        if filter:
            conditions = []
            for key, value in filter.items():
                import json
                conditions.append(f"metadata @> '{json.dumps({key: value})}'::jsonb")
            where_clause = "WHERE " + " AND ".join(conditions)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) as score
                FROM {self.config.table_name}
                {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, query_embedding, top_k)
        
        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                score=float(row["score"]),
                metadata=dict(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete documents from PGVector."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.config.table_name}
                WHERE id = ANY($1)
            """, ids)
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        """Get documents by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT id, content, embedding, metadata
                FROM {self.config.table_name}
                WHERE id = ANY($1)
            """, ids)
        
        return [
            Document(
                id=row["id"],
                content=row["content"],
                embedding=list(row["embedding"]) if row["embedding"] else None,
                metadata=dict(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]
    
    async def clear(self) -> bool:
        """Clear all documents."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {self.config.table_name}")
        return True


__all__ = ["PGVectorStore", "PGVectorConfig"]
