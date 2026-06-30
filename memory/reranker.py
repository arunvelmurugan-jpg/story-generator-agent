"""
RAG Reranker for PHTN.AI Sub-Agent Framework

Implements document reranking for RAG pipelines:
- Cross-Encoder reranking
- Cohere Rerank API
- BM25 scoring
- Reciprocal Rank Fusion
- Custom scoring functions
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import re
from collections import Counter

logger = logging.getLogger(__name__)


class RerankerProvider(str, Enum):
    """Reranker provider types."""
    COHERE = "cohere"
    CROSS_ENCODER = "cross_encoder"
    BM25 = "bm25"
    RECIPROCAL_RANK_FUSION = "rrf"
    CUSTOM = "custom"
    HYBRID = "hybrid"


@dataclass
class RerankerConfig:
    """Configuration for reranker."""
    enabled: bool = True
    provider: RerankerProvider = RerankerProvider.BM25
    model: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.0
    use_query_expansion: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    cohere_api_key: Optional[str] = None


@dataclass
class RankedDocument:
    """A ranked document with score."""
    content: str
    score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class RerankerResult:
    """Result of reranking operation."""
    documents: List[RankedDocument]
    query: str
    provider: str
    total_candidates: int
    returned_count: int
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "query": self.query,
            "provider": self.provider,
            "total_candidates": self.total_candidates,
            "returned_count": self.returned_count,
            "processing_time_ms": self.processing_time_ms,
        }
    
    def get_contents(self) -> List[str]:
        """Get just the document contents."""
        return [d.content for d in self.documents]


class BM25Scorer:
    """BM25 scoring implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._initialized = False
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 on document corpus.
        
        Args:
            documents: List of documents
        """
        self._corpus = [self._tokenize(doc) for doc in documents]
        self._doc_lengths = [len(doc) for doc in self._corpus]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0
        
        self._doc_freqs = {}
        for doc in self._corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
        
        n_docs = len(self._corpus)
        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        self._initialized = True
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a document.
        
        Args:
            query: Query string
            doc_idx: Document index
            
        Returns:
            BM25 score
        """
        if not self._initialized or doc_idx >= len(self._corpus):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc = self._corpus[doc_idx]
        doc_length = self._doc_lengths[doc_idx]
        
        term_freqs = Counter(doc)
        
        score = 0.0
        for term in query_terms:
            if term not in self._idf:
                continue
            
            tf = term_freqs.get(term, 0)
            idf = self._idf[term]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
            
            score += idf * numerator / denominator
        
        return score
    
    def score_all(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of (doc_idx, score) tuples sorted by score descending
        """
        scores = []
        for idx in range(len(self._corpus)):
            score = self.score(query, idx)
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class Reranker:
    """
    Document reranker for RAG pipelines.
    
    Features:
    - Multiple reranking strategies
    - BM25 scoring
    - Reciprocal Rank Fusion
    - Cohere API integration
    - Custom scoring functions
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize reranker.
        
        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()
        self._bm25: Optional[BM25Scorer] = None
        self._custom_scorer: Optional[Callable] = None
        
        logger.debug(f"Reranker initialized with provider: {self.config.provider.value}")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> RerankerResult:
        """
        Rerank documents for a query.
        
        Args:
            query: Search query
            documents: List of document contents
            metadata: Optional metadata for each document
            
        Returns:
            RerankerResult with ranked documents
        """
        import time
        start_time = time.time()
        
        if not self.config.enabled or not documents:
            ranked = [
                RankedDocument(
                    content=doc,
                    score=1.0 - (i * 0.1),
                    original_rank=i,
                    new_rank=i,
                    metadata=metadata[i] if metadata and i < len(metadata) else {},
                )
                for i, doc in enumerate(documents)
            ]
            return RerankerResult(
                documents=ranked[:self.config.top_k],
                query=query,
                provider="passthrough",
                total_candidates=len(documents),
                returned_count=min(len(documents), self.config.top_k),
            )
        
        expanded_query = query
        if self.config.use_query_expansion:
            expanded_query = self._expand_query(query)
        
        provider = self.config.provider
        
        if provider == RerankerProvider.BM25:
            ranked = await self._rerank_bm25(expanded_query, documents, metadata)
        elif provider == RerankerProvider.COHERE:
            ranked = await self._rerank_cohere(expanded_query, documents, metadata)
        elif provider == RerankerProvider.CROSS_ENCODER:
            ranked = await self._rerank_cross_encoder(expanded_query, documents, metadata)
        elif provider == RerankerProvider.RECIPROCAL_RANK_FUSION:
            ranked = await self._rerank_rrf(expanded_query, documents, metadata)
        elif provider == RerankerProvider.HYBRID:
            ranked = await self._rerank_hybrid(expanded_query, documents, metadata)
        elif provider == RerankerProvider.CUSTOM and self._custom_scorer:
            ranked = await self._rerank_custom(expanded_query, documents, metadata)
        else:
            ranked = await self._rerank_bm25(expanded_query, documents, metadata)
        
        filtered = [d for d in ranked if d.score >= self.config.min_score]
        top_k = filtered[:self.config.top_k]
        
        for i, doc in enumerate(top_k):
            doc.new_rank = i
        
        processing_time = (time.time() - start_time) * 1000
        
        return RerankerResult(
            documents=top_k,
            query=query,
            provider=provider.value,
            total_candidates=len(documents),
            returned_count=len(top_k),
            processing_time_ms=processing_time,
        )
    
    async def _rerank_bm25(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Rerank using BM25."""
        self._bm25 = BM25Scorer(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self._bm25.fit(documents)
        
        scores = self._bm25.score_all(query)
        
        max_score = max(s for _, s in scores) if scores else 1.0
        
        ranked = []
        for new_rank, (doc_idx, score) in enumerate(scores):
            normalized_score = score / max_score if max_score > 0 else 0.0
            ranked.append(RankedDocument(
                content=documents[doc_idx],
                score=normalized_score,
                original_rank=doc_idx,
                new_rank=new_rank,
                metadata=metadata[doc_idx] if metadata and doc_idx < len(metadata) else {},
            ))
        
        return ranked
    
    async def _rerank_cohere(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Rerank using Cohere API."""
        if not self.config.cohere_api_key:
            logger.warning("Cohere API key not configured, falling back to BM25")
            return await self._rerank_bm25(query, documents, metadata)
        
        try:
            import cohere
            
            client = cohere.Client(self.config.cohere_api_key)
            model = self.config.model or "rerank-english-v3.0"
            
            response = client.rerank(
                query=query,
                documents=documents,
                model=model,
                top_n=len(documents),
            )
            
            ranked = []
            for result in response.results:
                ranked.append(RankedDocument(
                    content=documents[result.index],
                    score=result.relevance_score,
                    original_rank=result.index,
                    new_rank=len(ranked),
                    metadata=metadata[result.index] if metadata and result.index < len(metadata) else {},
                ))
            
            return ranked
            
        except ImportError:
            logger.warning("Cohere package not installed, falling back to BM25")
            return await self._rerank_bm25(query, documents, metadata)
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}, falling back to BM25")
            return await self._rerank_bm25(query, documents, metadata)
    
    async def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Rerank using cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = self.config.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            model = CrossEncoder(model_name)
            
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)
            
            scored_docs = list(zip(range(len(documents)), scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            ranked = []
            for new_rank, (doc_idx, score) in enumerate(scored_docs):
                ranked.append(RankedDocument(
                    content=documents[doc_idx],
                    score=float(score),
                    original_rank=doc_idx,
                    new_rank=new_rank,
                    metadata=metadata[doc_idx] if metadata and doc_idx < len(metadata) else {},
                ))
            
            return ranked
            
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to BM25")
            return await self._rerank_bm25(query, documents, metadata)
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}, falling back to BM25")
            return await self._rerank_bm25(query, documents, metadata)
    
    async def _rerank_rrf(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Rerank using Reciprocal Rank Fusion."""
        bm25_ranked = await self._rerank_bm25(query, documents, metadata)
        
        semantic_scores = {}
        query_terms = set(query.lower().split())
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            semantic_scores[i] = overlap / len(query_terms) if query_terms else 0
        
        semantic_ranked = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        
        k = self.config.rrf_k
        rrf_scores = {}
        
        for rank, doc in enumerate(bm25_ranked):
            rrf_scores[doc.original_rank] = rrf_scores.get(doc.original_rank, 0) + 1 / (k + rank + 1)
        
        for rank, (doc_idx, _) in enumerate(semantic_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranked = []
        for new_rank, (doc_idx, score) in enumerate(sorted_docs):
            ranked.append(RankedDocument(
                content=documents[doc_idx],
                score=score,
                original_rank=doc_idx,
                new_rank=new_rank,
                metadata=metadata[doc_idx] if metadata and doc_idx < len(metadata) else {},
            ))
        
        return ranked
    
    async def _rerank_hybrid(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Hybrid reranking combining multiple methods."""
        bm25_ranked = await self._rerank_bm25(query, documents, metadata)
        
        combined_scores = {}
        for doc in bm25_ranked:
            combined_scores[doc.original_rank] = doc.score * 0.5
        
        query_terms = set(query.lower().split())
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            combined_scores[i] = combined_scores.get(i, 0) + overlap * 0.3
        
        for i, doc in enumerate(documents):
            length_score = min(len(doc) / 500, 1.0) * 0.1
            combined_scores[i] = combined_scores.get(i, 0) + length_score
        
        for i, doc in enumerate(documents):
            if any(term in doc.lower() for term in query_terms):
                combined_scores[i] = combined_scores.get(i, 0) + 0.1
        
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranked = []
        for new_rank, (doc_idx, score) in enumerate(sorted_docs):
            ranked.append(RankedDocument(
                content=documents[doc_idx],
                score=score,
                original_rank=doc_idx,
                new_rank=new_rank,
                metadata=metadata[doc_idx] if metadata and doc_idx < len(metadata) else {},
            ))
        
        return ranked
    
    async def _rerank_custom(
        self,
        query: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[RankedDocument]:
        """Rerank using custom scorer function."""
        if not self._custom_scorer:
            return await self._rerank_bm25(query, documents, metadata)
        
        scores = []
        for i, doc in enumerate(documents):
            score = self._custom_scorer(query, doc, metadata[i] if metadata else {})
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        ranked = []
        for new_rank, (doc_idx, score) in enumerate(scores):
            ranked.append(RankedDocument(
                content=documents[doc_idx],
                score=score,
                original_rank=doc_idx,
                new_rank=new_rank,
                metadata=metadata[doc_idx] if metadata and doc_idx < len(metadata) else {},
            ))
        
        return ranked
    
    def set_custom_scorer(self, scorer: Callable[[str, str, Dict], float]):
        """Set custom scoring function."""
        self._custom_scorer = scorer
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        synonyms = {
            "find": ["search", "locate", "get"],
            "create": ["make", "generate", "build"],
            "delete": ["remove", "erase", "clear"],
            "update": ["modify", "change", "edit"],
            "show": ["display", "list", "view"],
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, syns in synonyms.items():
            if term in query_lower:
                expanded_terms.extend(syns)
        
        return " ".join(expanded_terms)


def create_reranker(config_dict: Optional[Dict[str, Any]] = None) -> Reranker:
    """Create reranker from config dict."""
    if config_dict is None:
        config_dict = {}
    
    provider_str = config_dict.get("provider", "bm25")
    try:
        provider = RerankerProvider(provider_str)
    except ValueError:
        provider = RerankerProvider.BM25
    
    config = RerankerConfig(
        enabled=config_dict.get("enabled", True),
        provider=provider,
        model=config_dict.get("model"),
        top_k=config_dict.get("top_k", 5),
        min_score=config_dict.get("min_score", 0.0),
        use_query_expansion=config_dict.get("use_query_expansion", False),
        bm25_k1=config_dict.get("bm25_k1", 1.5),
        bm25_b=config_dict.get("bm25_b", 0.75),
        rrf_k=config_dict.get("rrf_k", 60),
        cohere_api_key=config_dict.get("cohere_api_key"),
    )
    
    return Reranker(config)
