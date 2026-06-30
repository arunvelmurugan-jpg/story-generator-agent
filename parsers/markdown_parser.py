"""
Markdown Output Parser

Parses LLM outputs in markdown format.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarkdownSection:
    """Represents a markdown section."""
    title: str
    level: int
    content: str
    subsections: List["MarkdownSection"]


class MarkdownOutputParser:
    """
    Parses LLM outputs in markdown format.
    
    Features:
    - Header extraction
    - Code block extraction
    - List parsing
    - Table parsing
    - Link extraction
    """
    
    def __init__(self):
        pass
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM."""
        return """Format your response using markdown:
- Use headers (## Section) to organize content
- Use bullet points for lists
- Use code blocks for code
- Use tables for structured data"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse markdown text into structured data."""
        return {
            "headers": self.extract_headers(text),
            "code_blocks": self.extract_code_blocks(text),
            "lists": self.extract_lists(text),
            "tables": self.extract_tables(text),
            "links": self.extract_links(text),
            "raw": text
        }
    
    def extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract headers from markdown."""
        headers = []
        pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            headers.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "position": match.start()
            })
        
        return headers
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown."""
        blocks = []
        pattern = r'```(\w*)\n([\s\S]*?)```'
        
        for match in re.finditer(pattern, text):
            blocks.append({
                "language": match.group(1) or "text",
                "code": match.group(2).strip()
            })
        
        inline_pattern = r'`([^`]+)`'
        for match in re.finditer(inline_pattern, text):
            if '```' not in match.group():
                blocks.append({
                    "language": "inline",
                    "code": match.group(1)
                })
        
        return blocks
    
    def extract_lists(self, text: str) -> List[Dict[str, Any]]:
        """Extract lists from markdown."""
        lists = []
        
        bullet_pattern = r'^[\-\*\+]\s+(.+)$'
        numbered_pattern = r'^\d+[\.\)]\s+(.+)$'
        
        current_list = []
        list_type = None
        
        for line in text.split('\n'):
            bullet_match = re.match(bullet_pattern, line)
            numbered_match = re.match(numbered_pattern, line)
            
            if bullet_match:
                if list_type == "numbered" and current_list:
                    lists.append({"type": "numbered", "items": current_list})
                    current_list = []
                list_type = "bullet"
                current_list.append(bullet_match.group(1))
            elif numbered_match:
                if list_type == "bullet" and current_list:
                    lists.append({"type": "bullet", "items": current_list})
                    current_list = []
                list_type = "numbered"
                current_list.append(numbered_match.group(1))
            elif current_list and line.strip() == "":
                lists.append({"type": list_type, "items": current_list})
                current_list = []
                list_type = None
        
        if current_list:
            lists.append({"type": list_type, "items": current_list})
        
        return lists
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown."""
        tables = []
        
        table_pattern = r'(\|[^\n]+\|\n)(\|[\-\:\|]+\|\n)((?:\|[^\n]+\|\n?)+)'
        
        for match in re.finditer(table_pattern, text):
            header_row = match.group(1).strip()
            data_rows = match.group(3).strip()
            
            headers = [h.strip() for h in header_row.split('|')[1:-1]]
            
            rows = []
            for row in data_rows.split('\n'):
                if row.strip():
                    cells = [c.strip() for c in row.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))
            
            tables.append({
                "headers": headers,
                "rows": rows
            })
        
        return tables
    
    def extract_links(self, text: str) -> List[Dict[str, str]]:
        """Extract links from markdown."""
        links = []
        
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        for match in re.finditer(link_pattern, text):
            links.append({
                "text": match.group(1),
                "url": match.group(2)
            })
        
        ref_pattern = r'\[([^\]]+)\]\[([^\]]+)\]'
        for match in re.finditer(ref_pattern, text):
            links.append({
                "text": match.group(1),
                "reference": match.group(2)
            })
        
        return links
    
    def extract_sections(self, text: str) -> List[MarkdownSection]:
        """Extract hierarchical sections from markdown."""
        headers = self.extract_headers(text)
        lines = text.split('\n')
        sections = []
        
        for i, header in enumerate(headers):
            start = header["position"]
            end = headers[i + 1]["position"] if i + 1 < len(headers) else len(text)
            content = text[start:end]
            
            header_line = re.match(r'^#+\s+.+$', content, re.MULTILINE)
            if header_line:
                content = content[header_line.end():].strip()
            
            sections.append(MarkdownSection(
                title=header["text"],
                level=header["level"],
                content=content,
                subsections=[]
            ))
        
        return sections


def create_markdown_parser() -> MarkdownOutputParser:
    """Factory function to create markdown parser."""
    return MarkdownOutputParser()


__all__ = ["MarkdownOutputParser", "MarkdownSection", "create_markdown_parser"]
