"""
Output Parsers Module - n8n Compatible

Provides output parsing strategies for LLM responses:
- StructuredOutputParser: Parse JSON/structured outputs
- AutoFixingOutputParser: Auto-fix malformed outputs
- ItemListOutputParser: Parse list outputs
- XMLOutputParser: Parse XML outputs
- RegexParser: Parse using regex patterns
- PydanticOutputParser: Parse to Pydantic models
- CommaSeparatedListParser: Parse comma-separated lists
- MarkdownOutputParser: Parse markdown outputs

Aligned with n8n's output parser nodes.
"""

from .structured import StructuredOutputParser, StructuredOutputConfig
from .auto_fixing import AutoFixingOutputParser, AutoFixingConfig
from .item_list import ItemListOutputParser, ItemListConfig
from .xml_parser import XMLOutputParser, XMLParserConfig
from .regex_parser import RegexParser, RegexParserConfig
from .pydantic_parser import PydanticOutputParser
from .markdown_parser import MarkdownOutputParser

__all__ = [
    "StructuredOutputParser",
    "StructuredOutputConfig",
    "AutoFixingOutputParser",
    "AutoFixingConfig",
    "ItemListOutputParser",
    "ItemListConfig",
    "XMLOutputParser",
    "XMLParserConfig",
    "RegexParser",
    "RegexParserConfig",
    "PydanticOutputParser",
    "MarkdownOutputParser",
]
