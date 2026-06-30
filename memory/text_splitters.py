"""
Text Splitters Module - n8n Compatible

Provides text chunking strategies for RAG pipelines:
- CharacterTextSplitter: Split by character count
- RecursiveCharacterTextSplitter: Smart recursive splitting
- TokenTextSplitter: Split by token count
- SemanticTextSplitter: Split by semantic boundaries
- MarkdownTextSplitter: Split markdown documents
- CodeTextSplitter: Split code files by language
- HTMLTextSplitter: Split HTML documents

Aligned with n8n's text splitter nodes for seamless migration.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class SplitterType(str, Enum):
    """Text splitter types."""
    CHARACTER = "character"
    RECURSIVE_CHARACTER = "recursive_character"
    TOKEN = "token"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    CODE = "code"
    HTML = "html"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    
    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class SplitterConfig:
    """Configuration for text splitters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    length_function: Callable[[str], int] = len
    keep_separator: bool = True
    add_start_index: bool = True
    strip_whitespace: bool = True


class BaseTextSplitter(ABC):
    """Base class for all text splitters."""
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        self._validate_config()
    
    def _validate_config(self):
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    
    @abstractmethod
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks."""
        pass
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split multiple documents."""
        result = []
        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            metadata = doc.get("metadata", {})
            chunks = self.split_text(text)
            for chunk in chunks:
                chunk.metadata.update(metadata)
                result.append({
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                })
        return result
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting size limits."""
        merged = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.config.length_function(split)
            
            if current_length + split_length > self.config.chunk_size:
                if current_chunk:
                    merged.append(separator.join(current_chunk))
                    
                    overlap_chunks = []
                    overlap_length = 0
                    for item in reversed(current_chunk):
                        item_len = self.config.length_function(item)
                        if overlap_length + item_len <= self.config.chunk_overlap:
                            overlap_chunks.insert(0, item)
                            overlap_length += item_len
                        else:
                            break
                    current_chunk = overlap_chunks
                    current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_length
        
        if current_chunk:
            merged.append(separator.join(current_chunk))
        
        return merged
    
    def _create_chunks(self, texts: List[str], original_text: str) -> List[TextChunk]:
        """Create TextChunk objects from text list."""
        chunks = []
        current_pos = 0
        
        for i, text in enumerate(texts):
            if self.config.strip_whitespace:
                text = text.strip()
            
            if not text:
                continue
            
            start_char = original_text.find(text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(text)
            
            chunk = TextChunk(
                content=text,
                index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                metadata={"chunk_index": len(chunks)}
            )
            chunks.append(chunk)
            current_pos = end_char
        
        return chunks


class CharacterTextSplitter(BaseTextSplitter):
    """
    Split text by character count.
    
    Simple splitter that divides text into fixed-size chunks
    with optional overlap. Similar to n8n's Character Text Splitter.
    """
    
    def __init__(
        self,
        separator: str = "\n\n",
        config: Optional[SplitterConfig] = None
    ):
        super().__init__(config)
        self.separator = separator
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by separator then merge into chunks."""
        if not text:
            return []
        
        if self.separator:
            splits = text.split(self.separator)
            if self.config.keep_separator and self.separator:
                splits = [s + self.separator for s in splits[:-1]] + [splits[-1]]
        else:
            splits = list(text)
        
        merged = self._merge_splits(splits, self.separator if self.config.keep_separator else "")
        return self._create_chunks(merged, text)


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursively split text using multiple separators.
    
    Tries to split on the most semantically meaningful boundaries first,
    falling back to smaller separators. This is the recommended splitter
    for most use cases. Equivalent to n8n's Recursive Character Text Splitter.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        config: Optional[SplitterConfig] = None
    ):
        super().__init__(config)
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Recursively split text."""
        if not text:
            return []
        
        final_chunks = self._split_text_recursive(text, self.separators)
        return self._create_chunks(final_chunks, text)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        good_splits = []
        for split in splits:
            if self.config.length_function(split) < self.config.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                if new_separators:
                    other_chunks = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks


class TokenTextSplitter(BaseTextSplitter):
    """
    Split text by token count.
    
    Uses tiktoken for accurate token counting with OpenAI models.
    Equivalent to n8n's Token Splitter node.
    """
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        config: Optional[SplitterConfig] = None
    ):
        super().__init__(config)
        self.encoding_name = encoding_name
        self.model_name = model_name
        self._tokenizer = None
    
    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                if self.model_name:
                    self._tokenizer = tiktoken.encoding_for_model(self.model_name)
                else:
                    self._tokenizer = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                logger.warning("tiktoken not installed, falling back to character splitting")
                self._tokenizer = None
        return self._tokenizer
    
    def _token_length(self, text: str) -> int:
        """Get token count for text."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text) // 4
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by token count."""
        if not text:
            return []
        
        tokenizer = self._get_tokenizer()
        if not tokenizer:
            splitter = CharacterTextSplitter(config=self.config)
            return splitter.split_text(text)
        
        tokens = tokenizer.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.config.chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            start_char = text.find(chunk_text) if chunks else 0
            if start_char == -1:
                start_char = chunks[-1].end_char if chunks else 0
            
            chunk = TextChunk(
                content=chunk_text,
                index=len(chunks),
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                token_count=len(chunk_tokens),
                metadata={"token_count": len(chunk_tokens)}
            )
            chunks.append(chunk)
            
            i += self.config.chunk_size - self.config.chunk_overlap
        
        return chunks


class SemanticTextSplitter(BaseTextSplitter):
    """
    Split text by semantic boundaries using embeddings.
    
    Groups semantically similar sentences together.
    Requires an embedding model for similarity computation.
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        breakpoint_threshold: float = 0.5,
        config: Optional[SplitterConfig] = None
    ):
        super().__init__(config)
        self.embedding_function = embedding_function
        self.breakpoint_threshold = breakpoint_threshold
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by semantic boundaries."""
        if not text:
            return []
        
        if not self.embedding_function:
            splitter = RecursiveCharacterTextSplitter(config=self.config)
            return splitter.split_text(text)
        
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._create_chunks(sentences, text)
        
        embeddings = self.embedding_function(sentences)
        
        breakpoints = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if similarity < self.breakpoint_threshold:
                breakpoints.append(i + 1)
        
        chunks_text = []
        start = 0
        for bp in breakpoints:
            chunk = " ".join(sentences[start:bp])
            if chunk:
                chunks_text.append(chunk)
            start = bp
        
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk:
                chunks_text.append(chunk)
        
        return self._create_chunks(chunks_text, text)


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """
    Split markdown documents by headers and structure.
    
    Preserves markdown structure while splitting.
    """
    
    MARKDOWN_SEPARATORS = [
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n##### ",
        "\n###### ",
        "\n\n",
        "\n",
        " ",
        ""
    ]
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        super().__init__(separators=self.MARKDOWN_SEPARATORS, config=config)
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split markdown text preserving structure."""
        chunks = super().split_text(text)
        
        for chunk in chunks:
            headers = re.findall(r'^(#{1,6})\s+(.+)$', chunk.content, re.MULTILINE)
            if headers:
                chunk.metadata["headers"] = [{"level": len(h[0]), "text": h[1]} for h in headers]
        
        return chunks


class CodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    Split code files by language-specific boundaries.
    
    Supports multiple programming languages with appropriate separators.
    """
    
    LANGUAGE_SEPARATORS = {
        "python": ["\nclass ", "\ndef ", "\n\ndef ", "\n\n", "\n", " ", ""],
        "javascript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ", "\n\n", "\n", " ", ""],
        "typescript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ", "\ninterface ", "\ntype ", "\n\n", "\n", " ", ""],
        "java": ["\npublic class ", "\nprivate class ", "\nclass ", "\npublic ", "\nprivate ", "\n\n", "\n", " ", ""],
        "go": ["\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\n\n", "\n", " ", ""],
        "rust": ["\nfn ", "\npub fn ", "\nimpl ", "\nstruct ", "\nenum ", "\n\n", "\n", " ", ""],
        "cpp": ["\nclass ", "\nvoid ", "\nint ", "\n\n", "\n", " ", ""],
        "csharp": ["\npublic class ", "\nprivate class ", "\nclass ", "\npublic ", "\nprivate ", "\n\n", "\n", " ", ""],
        "ruby": ["\nclass ", "\ndef ", "\nmodule ", "\n\n", "\n", " ", ""],
        "php": ["\nfunction ", "\nclass ", "\n\n", "\n", " ", ""],
        "sql": ["\nSELECT ", "\nINSERT ", "\nUPDATE ", "\nDELETE ", "\nCREATE ", "\nALTER ", "\n\n", "\n", " ", ""],
    }
    
    def __init__(
        self,
        language: str = "python",
        config: Optional[SplitterConfig] = None
    ):
        separators = self.LANGUAGE_SEPARATORS.get(
            language.lower(),
            RecursiveCharacterTextSplitter.DEFAULT_SEPARATORS
        )
        super().__init__(separators=separators, config=config)
        self.language = language
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split code text by language-specific boundaries."""
        chunks = super().split_text(text)
        
        for chunk in chunks:
            chunk.metadata["language"] = self.language
        
        return chunks


class HTMLTextSplitter(BaseTextSplitter):
    """
    Split HTML documents by tags and structure.
    
    Extracts text content while preserving document structure.
    """
    
    def __init__(
        self,
        tags_to_split: Optional[List[str]] = None,
        config: Optional[SplitterConfig] = None
    ):
        super().__init__(config)
        self.tags_to_split = tags_to_split or ["div", "p", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"]
    
    def _extract_text(self, html: str) -> str:
        """Extract text from HTML."""
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split HTML text by tags."""
        if not text:
            return []
        
        chunks_text = []
        
        for tag in self.tags_to_split:
            pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                extracted = self._extract_text(match)
                if extracted and len(extracted) > 10:
                    chunks_text.append(extracted)
        
        if not chunks_text:
            extracted = self._extract_text(text)
            splitter = RecursiveCharacterTextSplitter(config=self.config)
            return splitter.split_text(extracted)
        
        final_chunks = []
        for chunk_text in chunks_text:
            if self.config.length_function(chunk_text) > self.config.chunk_size:
                splitter = RecursiveCharacterTextSplitter(config=self.config)
                sub_chunks = splitter.split_text(chunk_text)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(TextChunk(
                    content=chunk_text,
                    index=len(final_chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata={"source": "html"}
                ))
        
        for i, chunk in enumerate(final_chunks):
            chunk.index = i
        
        return final_chunks


class SentenceTextSplitter(BaseTextSplitter):
    """
    Split text by sentences.
    
    Uses sentence boundary detection for natural splits.
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        super().__init__(config)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by sentences."""
        if not text:
            return []
        
        sentences = self._split_sentences(text)
        merged = self._merge_splits(sentences, " ")
        return self._create_chunks(merged, text)


class ParagraphTextSplitter(BaseTextSplitter):
    """
    Split text by paragraphs.
    
    Splits on double newlines to preserve paragraph structure.
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        super().__init__(config)
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by paragraphs."""
        if not text:
            return []
        
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        merged = self._merge_splits(paragraphs, "\n\n")
        return self._create_chunks(merged, text)


def create_text_splitter(
    splitter_type: SplitterType,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> BaseTextSplitter:
    """
    Factory function to create text splitters.
    
    Args:
        splitter_type: Type of splitter to create
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional splitter-specific arguments
    
    Returns:
        Configured text splitter instance
    """
    config = SplitterConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=kwargs.pop("keep_separator", True),
        strip_whitespace=kwargs.pop("strip_whitespace", True)
    )
    
    splitters = {
        SplitterType.CHARACTER: lambda: CharacterTextSplitter(
            separator=kwargs.get("separator", "\n\n"),
            config=config
        ),
        SplitterType.RECURSIVE_CHARACTER: lambda: RecursiveCharacterTextSplitter(
            separators=kwargs.get("separators"),
            config=config
        ),
        SplitterType.TOKEN: lambda: TokenTextSplitter(
            encoding_name=kwargs.get("encoding_name", "cl100k_base"),
            model_name=kwargs.get("model_name"),
            config=config
        ),
        SplitterType.SEMANTIC: lambda: SemanticTextSplitter(
            embedding_function=kwargs.get("embedding_function"),
            breakpoint_threshold=kwargs.get("breakpoint_threshold", 0.5),
            config=config
        ),
        SplitterType.MARKDOWN: lambda: MarkdownTextSplitter(config=config),
        SplitterType.CODE: lambda: CodeTextSplitter(
            language=kwargs.get("language", "python"),
            config=config
        ),
        SplitterType.HTML: lambda: HTMLTextSplitter(
            tags_to_split=kwargs.get("tags_to_split"),
            config=config
        ),
        SplitterType.SENTENCE: lambda: SentenceTextSplitter(config=config),
        SplitterType.PARAGRAPH: lambda: ParagraphTextSplitter(config=config),
    }
    
    if splitter_type not in splitters:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    return splitters[splitter_type]()


__all__ = [
    "SplitterType",
    "TextChunk",
    "SplitterConfig",
    "BaseTextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "SemanticTextSplitter",
    "MarkdownTextSplitter",
    "CodeTextSplitter",
    "HTMLTextSplitter",
    "SentenceTextSplitter",
    "ParagraphTextSplitter",
    "create_text_splitter",
]
