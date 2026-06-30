"""
XML Output Parser

Parses LLM outputs in XML format.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class XMLParserConfig:
    """Configuration for XML parser."""
    tags: List[str] = None
    root_tag: str = "response"
    extract_from_text: bool = True
    include_attributes: bool = True


class XMLOutputParser:
    """
    Parses LLM outputs in XML format.
    
    Features:
    - Tag-based extraction
    - Attribute parsing
    - Nested element support
    - Auto-extraction from mixed content
    """
    
    def __init__(self, config: Optional[XMLParserConfig] = None):
        self.config = config or XMLParserConfig()
        if self.config.tags is None:
            self.config.tags = []
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM."""
        tags_example = "\n".join([f"  <{tag}>value</{tag}>" for tag in self.config.tags])
        return f"""Return your response in XML format:
<{self.config.root_tag}>
{tags_example}
</{self.config.root_tag}>"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse XML text into dictionary."""
        xml_str = self._extract_xml(text) if self.config.extract_from_text else text
        
        try:
            root = ET.fromstring(xml_str)
            return self._element_to_dict(root)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}, attempting tag extraction")
            return self._extract_tags(text)
    
    def _extract_xml(self, text: str) -> str:
        """Extract XML from text."""
        xml_pattern = rf'<{self.config.root_tag}[\s\S]*?</{self.config.root_tag}>'
        match = re.search(xml_pattern, text)
        if match:
            return match.group()
        
        code_block = re.search(r'```xml\s*([\s\S]*?)\s*```', text)
        if code_block:
            return code_block.group(1)
        
        return text.strip()
    
    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        if self.config.include_attributes and element.attrib:
            result["@attributes"] = dict(element.attrib)
        
        for child in element:
            child_data = self._element_to_dict(child)
            
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data if child_data else child.text)
            else:
                result[child.tag] = child_data if child_data else child.text
        
        if not result and element.text:
            return element.text.strip()
        
        return result
    
    def _extract_tags(self, text: str) -> Dict[str, Any]:
        """Extract specific tags from text."""
        result = {}
        
        for tag in self.config.tags:
            pattern = rf'<{tag}[^>]*>([\s\S]*?)</{tag}>'
            matches = re.findall(pattern, text)
            
            if len(matches) == 1:
                result[tag] = matches[0].strip()
            elif len(matches) > 1:
                result[tag] = [m.strip() for m in matches]
        
        return result
    
    def parse_tags(self, text: str, tags: List[str]) -> Dict[str, str]:
        """Parse specific tags from text."""
        result = {}
        for tag in tags:
            pattern = rf'<{tag}>([\s\S]*?)</{tag}>'
            match = re.search(pattern, text)
            if match:
                result[tag] = match.group(1).strip()
        return result


def create_xml_parser(
    tags: Optional[List[str]] = None,
    root_tag: str = "response",
    **kwargs
) -> XMLOutputParser:
    """Factory function to create XML parser."""
    config = XMLParserConfig(tags=tags or [], root_tag=root_tag, **kwargs)
    return XMLOutputParser(config)


__all__ = ["XMLOutputParser", "XMLParserConfig", "create_xml_parser"]
