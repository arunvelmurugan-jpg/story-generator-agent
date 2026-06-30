"""
Vector Stores Module - n8n Compatible

Supports multiple vector database providers:
- Pinecone
- Chroma
- PGVector (PostgreSQL)
- Weaviate
- Qdrant
- Milvus
- In-Memory

Aligned with n8n's vector store nodes.
"""

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult
from .pinecone import PineconeVectorStore
from .chroma import ChromaVectorStore
from .pgvector import PGVectorStore
from .weaviate import WeaviateVectorStore
from .qdrant import QdrantVectorStore
from .memory import InMemoryVectorStore

__all__ = [
    "BaseVectorStore",
    "VectorStoreConfig",
    "Document",
    "SearchResult",
    "PineconeVectorStore",
    "ChromaVectorStore",
    "PGVectorStore",
    "WeaviateVectorStore",
    "QdrantVectorStore",
    "InMemoryVectorStore",
]
