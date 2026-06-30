"""
Context Compression for PHTN.AI Sub-Agent Framework

Implements context compression strategies to optimize LLM context window usage:
- Importance-weighted summarization
- Semantic deduplication
- Token-aware truncation
- Hierarchical compression
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class CompressionStrategy(str, Enum):
    """Context compression strategies."""
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    SEMANTIC_DEDUP = "semantic_dedup"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    enabled: bool = True
    strategy: CompressionStrategy = CompressionStrategy.IMPORTANCE_WEIGHTED
    max_tokens: int = 4000
    target_compression_ratio: float = 0.5
    preserve_recent_messages: int = 3
    preserve_system_prompt: bool = True
    importance_threshold: float = 0.3
    enable_semantic_dedup: bool = True
    dedup_similarity_threshold: float = 0.9


@dataclass
class CompressedContext:
    """Result of context compression."""
    content: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: str
    preserved_items: List[str] = field(default_factory=list)
    removed_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "strategy_used": self.strategy_used,
            "preserved_count": len(self.preserved_items),
            "removed_count": len(self.removed_items),
            "metadata": self.metadata,
        }


class ContextCompressor:
    """
    Compresses context to fit within token limits.
    
    Features:
    - Multiple compression strategies
    - Importance scoring
    - Semantic deduplication
    - Token-aware processing
    """
    
    IMPORTANCE_KEYWORDS = {
        "high": [
            "important", "critical", "must", "required", "essential",
            "key", "main", "primary", "core", "fundamental",
            "error", "warning", "exception", "fail", "issue",
        ],
        "medium": [
            "should", "recommend", "suggest", "consider", "note",
            "example", "instance", "case", "scenario",
        ],
        "low": [
            "also", "additionally", "furthermore", "moreover",
            "however", "although", "nevertheless",
        ],
    }
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize context compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        logger.debug(f"ContextCompressor initialized with strategy: {self.config.strategy.value}")
    
    def compress(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> CompressedContext:
        """
        Compress context to fit within token limits.
        
        Args:
            messages: Conversation messages
            documents: Optional retrieved documents
            max_tokens: Override max tokens
            
        Returns:
            CompressedContext
        """
        if not self.config.enabled:
            combined = self._combine_content(messages, documents)
            return CompressedContext(
                content=combined,
                original_tokens=self._estimate_tokens(combined),
                compressed_tokens=self._estimate_tokens(combined),
                compression_ratio=1.0,
                strategy_used="none",
            )
        
        max_tokens = max_tokens or self.config.max_tokens
        strategy = self.config.strategy
        
        if strategy == CompressionStrategy.TRUNCATE:
            return self._compress_truncate(messages, documents, max_tokens)
        elif strategy == CompressionStrategy.SUMMARIZE:
            return self._compress_summarize(messages, documents, max_tokens)
        elif strategy == CompressionStrategy.IMPORTANCE_WEIGHTED:
            return self._compress_importance_weighted(messages, documents, max_tokens)
        elif strategy == CompressionStrategy.SEMANTIC_DEDUP:
            return self._compress_semantic_dedup(messages, documents, max_tokens)
        elif strategy == CompressionStrategy.HIERARCHICAL:
            return self._compress_hierarchical(messages, documents, max_tokens)
        elif strategy == CompressionStrategy.HYBRID:
            return self._compress_hybrid(messages, documents, max_tokens)
        else:
            return self._compress_truncate(messages, documents, max_tokens)
    
    def _compress_truncate(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Simple truncation strategy."""
        combined = self._combine_content(messages, documents)
        original_tokens = self._estimate_tokens(combined)
        
        if original_tokens <= max_tokens:
            return CompressedContext(
                content=combined,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used="truncate",
            )
        
        truncated = self._truncate_to_tokens(combined, max_tokens)
        compressed_tokens = self._estimate_tokens(truncated)
        
        return CompressedContext(
            content=truncated,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used="truncate",
        )
    
    def _compress_summarize(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Summarization-based compression."""
        preserved = []
        summarized_parts = []
        
        if self.config.preserve_system_prompt:
            for msg in messages:
                if msg.get("role") == "system":
                    preserved.append(msg.get("content", ""))
                    break
        
        recent_count = self.config.preserve_recent_messages
        if recent_count > 0 and len(messages) > recent_count:
            recent_messages = messages[-recent_count:]
            older_messages = messages[:-recent_count]
            
            for msg in recent_messages:
                if msg.get("role") != "system":
                    preserved.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            
            older_content = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in older_messages
                if msg.get("role") != "system"
            ])
            
            if older_content:
                summary = self._extract_key_points(older_content)
                summarized_parts.append(f"[Previous conversation summary: {summary}]")
        else:
            for msg in messages:
                if msg.get("role") != "system":
                    preserved.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        
        if documents:
            doc_summary = self._summarize_documents(documents)
            summarized_parts.append(f"[Retrieved context: {doc_summary}]")
        
        combined = "\n\n".join(preserved + summarized_parts)
        original_tokens = self._estimate_tokens(self._combine_content(messages, documents))
        compressed_tokens = self._estimate_tokens(combined)
        
        return CompressedContext(
            content=combined,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used="summarize",
            preserved_items=preserved,
        )
    
    def _compress_importance_weighted(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Importance-weighted compression."""
        scored_items: List[Tuple[float, str, str]] = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            if role == "system" and self.config.preserve_system_prompt:
                score = 1.0
            else:
                score = self._calculate_importance(content, role)
            
            scored_items.append((score, role, content))
        
        if documents:
            for doc in documents:
                score = self._calculate_importance(doc, "document")
                scored_items.append((score, "document", doc))
        
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        preserved = []
        removed = []
        current_tokens = 0
        
        for score, role, content in scored_items:
            item_tokens = self._estimate_tokens(content)
            
            if current_tokens + item_tokens <= max_tokens:
                if role == "document":
                    preserved.append(f"[Context: {content}]")
                else:
                    preserved.append(f"{role}: {content}")
                current_tokens += item_tokens
            else:
                removed.append(content[:50] + "...")
        
        combined = "\n\n".join(preserved)
        original_tokens = self._estimate_tokens(self._combine_content(messages, documents))
        
        return CompressedContext(
            content=combined,
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used="importance_weighted",
            preserved_items=preserved,
            removed_items=removed,
            metadata={"threshold": self.config.importance_threshold},
        )
    
    def _compress_semantic_dedup(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Semantic deduplication compression."""
        all_content = []
        
        for msg in messages:
            all_content.append({
                "type": "message",
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        
        if documents:
            for doc in documents:
                all_content.append({
                    "type": "document",
                    "role": "document",
                    "content": doc,
                })
        
        unique_content = self._remove_duplicates(all_content)
        
        preserved = []
        for item in unique_content:
            if item["type"] == "message":
                preserved.append(f"{item['role']}: {item['content']}")
            else:
                preserved.append(f"[Context: {item['content']}]")
        
        combined = "\n\n".join(preserved)
        
        if self._estimate_tokens(combined) > max_tokens:
            combined = self._truncate_to_tokens(combined, max_tokens)
        
        original_tokens = self._estimate_tokens(self._combine_content(messages, documents))
        compressed_tokens = self._estimate_tokens(combined)
        
        return CompressedContext(
            content=combined,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used="semantic_dedup",
            metadata={
                "original_items": len(all_content),
                "unique_items": len(unique_content),
            },
        )
    
    def _compress_hierarchical(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Hierarchical compression with multiple levels."""
        system_prompt = ""
        recent_messages = []
        older_messages = []
        doc_content = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif len(recent_messages) < self.config.preserve_recent_messages:
                recent_messages.append(msg)
            else:
                older_messages.append(msg)
        
        if documents:
            doc_content = documents
        
        parts = []
        current_tokens = 0
        
        if system_prompt and self.config.preserve_system_prompt:
            parts.append(f"system: {system_prompt}")
            current_tokens += self._estimate_tokens(system_prompt)
        
        for msg in recent_messages:
            content = f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            parts.append(content)
            current_tokens += self._estimate_tokens(content)
        
        remaining_tokens = max_tokens - current_tokens
        
        if older_messages and remaining_tokens > 100:
            older_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in older_messages
            ])
            summary = self._extract_key_points(older_text)
            summary_tokens = self._estimate_tokens(summary)
            
            if summary_tokens <= remaining_tokens * 0.3:
                parts.insert(1 if system_prompt else 0, f"[Earlier context: {summary}]")
                current_tokens += summary_tokens
                remaining_tokens -= summary_tokens
        
        if doc_content and remaining_tokens > 100:
            doc_text = "\n".join(doc_content)
            doc_summary = self._summarize_documents(doc_content)
            doc_tokens = self._estimate_tokens(doc_summary)
            
            if doc_tokens <= remaining_tokens:
                parts.append(f"[Retrieved information: {doc_summary}]")
                current_tokens += doc_tokens
        
        combined = "\n\n".join(parts)
        original_tokens = self._estimate_tokens(self._combine_content(messages, documents))
        
        return CompressedContext(
            content=combined,
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            compression_ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy_used="hierarchical",
        )
    
    def _compress_hybrid(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
        max_tokens: int,
    ) -> CompressedContext:
        """Hybrid compression combining multiple strategies."""
        dedup_result = self._compress_semantic_dedup(messages, documents, max_tokens * 2)
        
        dedup_messages = []
        dedup_docs = []
        for line in dedup_result.content.split("\n\n"):
            if line.startswith("[Context:"):
                dedup_docs.append(line[10:-1])
            elif ": " in line:
                role, content = line.split(": ", 1)
                dedup_messages.append({"role": role, "content": content})
        
        importance_result = self._compress_importance_weighted(
            dedup_messages or messages,
            dedup_docs or documents,
            max_tokens,
        )
        
        return CompressedContext(
            content=importance_result.content,
            original_tokens=dedup_result.original_tokens,
            compressed_tokens=importance_result.compressed_tokens,
            compression_ratio=importance_result.compression_ratio,
            strategy_used="hybrid",
            preserved_items=importance_result.preserved_items,
            removed_items=importance_result.removed_items,
            metadata={
                "dedup_ratio": dedup_result.compression_ratio,
                "importance_ratio": importance_result.compression_ratio,
            },
        )
    
    def _calculate_importance(self, content: str, role: str) -> float:
        """Calculate importance score for content."""
        score = 0.5
        
        role_weights = {
            "system": 1.0,
            "user": 0.8,
            "assistant": 0.6,
            "document": 0.7,
        }
        score *= role_weights.get(role, 0.5)
        
        content_lower = content.lower()
        
        for keyword in self.IMPORTANCE_KEYWORDS["high"]:
            if keyword in content_lower:
                score += 0.1
        
        for keyword in self.IMPORTANCE_KEYWORDS["medium"]:
            if keyword in content_lower:
                score += 0.05
        
        if "?" in content:
            score += 0.1
        
        if re.search(r'\b\d+\b', content):
            score += 0.05
        
        return min(1.0, score)
    
    def _remove_duplicates(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove semantically similar items."""
        unique = []
        seen_content = []
        
        for item in items:
            content = item["content"].lower().strip()
            
            is_duplicate = False
            for seen in seen_content:
                similarity = self._calculate_similarity(content, seen)
                if similarity >= self.config.dedup_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_content.append(content)
        
        return unique
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _extract_key_points(self, text: str) -> str:
        """Extract key points from text."""
        sentences = re.split(r'[.!?]+', text)
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                score = self._calculate_importance(sentence, "text")
                scored_sentences.append((score, sentence))
        
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        top_sentences = [s for _, s in scored_sentences[:3]]
        return ". ".join(top_sentences) + "." if top_sentences else ""
    
    def _summarize_documents(self, documents: List[str]) -> str:
        """Summarize multiple documents."""
        all_text = " ".join(documents)
        return self._extract_key_points(all_text)
    
    def _combine_content(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[List[str]],
    ) -> str:
        """Combine messages and documents into single string."""
        parts = []
        
        for msg in messages:
            parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        
        if documents:
            for doc in documents:
                parts.append(f"[Document: {doc}]")
        
        return "\n\n".join(parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


def create_context_compressor(
    config_dict: Optional[Dict[str, Any]] = None,
) -> ContextCompressor:
    """Create context compressor from config dict."""
    if config_dict is None:
        config_dict = {}
    
    strategy_str = config_dict.get("strategy", "importance_weighted")
    try:
        strategy = CompressionStrategy(strategy_str)
    except ValueError:
        strategy = CompressionStrategy.IMPORTANCE_WEIGHTED
    
    config = CompressionConfig(
        enabled=config_dict.get("enabled", True),
        strategy=strategy,
        max_tokens=config_dict.get("max_tokens", 4000),
        target_compression_ratio=config_dict.get("target_compression_ratio", 0.5),
        preserve_recent_messages=config_dict.get("preserve_recent_messages", 3),
        preserve_system_prompt=config_dict.get("preserve_system_prompt", True),
        importance_threshold=config_dict.get("importance_threshold", 0.3),
        enable_semantic_dedup=config_dict.get("enable_semantic_dedup", True),
        dedup_similarity_threshold=config_dict.get("dedup_similarity_threshold", 0.9),
    )
    
    return ContextCompressor(config)
