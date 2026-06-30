"""
Item List Output Parser

Parses LLM outputs into lists of items.
Equivalent to n8n's Item List Output Parser node.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ItemListConfig:
    """Configuration for item list parser."""
    separator: str = "\n"
    numbered: bool = True
    strip_whitespace: bool = True
    remove_empty: bool = True
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_prefix: Optional[str] = None
    item_suffix: Optional[str] = None


class ItemListOutputParser:
    """
    Parses LLM outputs into lists of items.
    
    Features:
    - Multiple separator support
    - Numbered list parsing
    - Bullet point parsing
    - Item count validation
    - Prefix/suffix handling
    """
    
    NUMBERED_PATTERNS = [
        r'^\d+[\.\)]\s*',
        r'^[\-\*\•]\s*',
        r'^[a-zA-Z][\.\)]\s*',
    ]
    
    def __init__(self, config: Optional[ItemListConfig] = None):
        self.config = config or ItemListConfig()
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM."""
        if self.config.numbered:
            return """Return a numbered list of items, one per line:
1. First item
2. Second item
3. Third item"""
        else:
            return f"""Return a list of items separated by '{self.config.separator}'.
Each item should be on its own line."""
    
    def parse(self, text: str) -> List[str]:
        """Parse text into a list of items."""
        items = self._split_items(text)
        
        if self.config.numbered:
            items = self._remove_numbering(items)
        
        if self.config.strip_whitespace:
            items = [item.strip() for item in items]
        
        if self.config.remove_empty:
            items = [item for item in items if item]
        
        if self.config.item_prefix:
            items = [
                item[len(self.config.item_prefix):] if item.startswith(self.config.item_prefix) else item
                for item in items
            ]
        
        if self.config.item_suffix:
            items = [
                item[:-len(self.config.item_suffix)] if item.endswith(self.config.item_suffix) else item
                for item in items
            ]
        
        self._validate_count(items)
        
        return items
    
    def _split_items(self, text: str) -> List[str]:
        """Split text into items."""
        if self.config.separator == "\n":
            return text.split("\n")
        elif self.config.separator == ",":
            return self._smart_comma_split(text)
        else:
            return text.split(self.config.separator)
    
    def _smart_comma_split(self, text: str) -> List[str]:
        """Split by comma, respecting quoted strings."""
        items = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in text:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current += char
            elif char == "," and not in_quotes:
                items.append(current)
                current = ""
            else:
                current += char
        
        if current:
            items.append(current)
        
        return items
    
    def _remove_numbering(self, items: List[str]) -> List[str]:
        """Remove numbering/bullets from items."""
        cleaned = []
        for item in items:
            clean_item = item
            for pattern in self.NUMBERED_PATTERNS:
                clean_item = re.sub(pattern, "", clean_item)
            cleaned.append(clean_item)
        return cleaned
    
    def _validate_count(self, items: List[str]) -> None:
        """Validate item count."""
        if self.config.min_items and len(items) < self.config.min_items:
            raise ValueError(f"Expected at least {self.config.min_items} items, got {len(items)}")
        if self.config.max_items and len(items) > self.config.max_items:
            raise ValueError(f"Expected at most {self.config.max_items} items, got {len(items)}")
    
    def parse_to_dicts(
        self,
        text: str,
        key_name: str = "item"
    ) -> List[Dict[str, str]]:
        """Parse to list of dictionaries."""
        items = self.parse(text)
        return [{key_name: item, "index": i} for i, item in enumerate(items)]


class CommaSeparatedListParser(ItemListOutputParser):
    """Parser for comma-separated lists."""
    
    def __init__(self):
        super().__init__(ItemListConfig(separator=",", numbered=False))
    
    def get_format_instructions(self) -> str:
        return "Return items as a comma-separated list: item1, item2, item3"


class BulletListParser(ItemListOutputParser):
    """Parser for bullet point lists."""
    
    def __init__(self):
        super().__init__(ItemListConfig(separator="\n", numbered=True))
    
    def get_format_instructions(self) -> str:
        return """Return items as a bullet list:
- First item
- Second item
- Third item"""


class NumberedListParser(ItemListOutputParser):
    """Parser for numbered lists."""
    
    def __init__(self):
        super().__init__(ItemListConfig(separator="\n", numbered=True))
    
    def get_format_instructions(self) -> str:
        return """Return items as a numbered list:
1. First item
2. Second item
3. Third item"""


def create_item_list_parser(
    separator: str = "\n",
    numbered: bool = True,
    **kwargs
) -> ItemListOutputParser:
    """Factory function to create item list parser."""
    config = ItemListConfig(separator=separator, numbered=numbered, **kwargs)
    return ItemListOutputParser(config)


__all__ = [
    "ItemListOutputParser",
    "ItemListConfig",
    "CommaSeparatedListParser",
    "BulletListParser",
    "NumberedListParser",
    "create_item_list_parser"
]
