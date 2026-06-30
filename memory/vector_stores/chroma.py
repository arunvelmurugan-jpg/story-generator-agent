"""
Chroma Vector Store

Local/embedded vector database.
Equivalent to n8n's Chroma Vector Store node.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import BaseVectorStore, VectorStoreConfig, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ChromaConfig(VectorStoreConfig):
    """Chroma-specific configuration."""
    persist_directory: Optional[str] = None
    host: Optional[str] = None
    port: int = 8000
    ssl: bool = False
    headers: Optional[Dict[str, str]] = None


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store implementation.
    
    Features:
    - Local/embedded mode
    - Client/server mode
    - Persistent storage
    - Metadata filtering
    """
    
    def __init__(self, config: ChromaConfig):
        super().__init__(config)
        self.config: ChromaConfig = config
        self._client = None
        self._collection = None
    
    def _get_client(self):
        """Get or create Chroma client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.config.host:
                    self._client = chromadb.HttpClient(
                        host=self.config.host,
                        port=self.config.port,
                        ssl=self.config.ssl,
                        headers=self.config.headers
                    )
                elif self.config.persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=self.config.persist_directory
                    )
                else:
                    self._client = chromadb.Client()
                
            except ImportError:
                raise ImportError("chromadb required: pip install chromadb")
        return self._client
    
    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.metric}
            )
        return self._collection
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Chroma."""
        collection = self._get_collection()
        
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        if embeddings and len(embeddings) == len(documents):
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        
        return ids
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Chroma for similar documents."""
        collection = self._get_collection()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        search_results = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results.get("documents") else "",
                    score=1 - results["distances"][0][i] if results.get("distances") else 0,
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {}
                ))
        
        return search_results
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete documents from Chroma."""
        collection = self._get_collection()
        collection.delete(ids=ids)
        return True
    
    async def get(self, ids: List[str]) -> List[Document]:
        """Get documents by ID."""
        collection = self._get_collection()
        results = collection.get(ids=ids)
        
        documents = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"]):
                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][i] if results.get("documents") else "",
                    metadata=results["metadatas"][i] if results.get("metadatas") else {}
                ))
        
        return documents
    
    async def clear(self) -> bool:
        """Clear all documents."""
        client = self._get_client()
        client.delete_collection(self.config.collection_name)
        self._collection = None
        return True


__all__ = ["ChromaVectorStore", "ChromaConfig"]
