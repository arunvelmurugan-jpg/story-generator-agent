"""
Memory Management for PHTN.AI Sub-Agent Framework

Provides memory management with multiple backends:
- Short-term (session) memory
- Long-term persistent memory (Redis, PostgreSQL)
- Semantic (vector) memory (Pinecone, Milvus, Qdrant, Weaviate, ChromaDB)
- Episodic memory
- Context compression
- RAG reranking
"""

from .manager import MemoryManager
from .vector_store import (
    VectorStoreConfig,
    VectorStoreProvider,
    VectorDocument,
    BaseVectorStore,
    InMemoryVectorStore,
    PineconeVectorStore,
    MilvusVectorStore,
    QdrantVectorStore,
    create_vector_store,
)
from .embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    CohereEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    LocalEmbeddingProvider,
    create_embedding_provider,
)
from .context_compression import (
    CompressionConfig,
    CompressionStrategy,
    CompressedContext,
    ContextCompressor,
    create_context_compressor,
)
from .reranker import (
    RerankerConfig,
    RerankerProvider,
    RankedDocument,
    RerankerResult,
    Reranker,
    BM25Scorer,
    create_reranker,
)

__all__ = [
    "MemoryManager",
    "VectorStoreConfig",
    "VectorStoreProvider",
    "VectorDocument",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "MilvusVectorStore",
    "QdrantVectorStore",
    "create_vector_store",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "LocalEmbeddingProvider",
    "create_embedding_provider",
    "CompressionConfig",
    "CompressionStrategy",
    "CompressedContext",
    "ContextCompressor",
    "create_context_compressor",
    "RerankerConfig",
    "RerankerProvider",
    "RankedDocument",
    "RerankerResult",
    "Reranker",
    "BM25Scorer",
    "create_reranker",
]
