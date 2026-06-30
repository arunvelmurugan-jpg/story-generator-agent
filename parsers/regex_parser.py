"""
Regex Output Parser
"""

import re
import logging
from typing import Any, Dict, List, Optional, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegexParserConfig:
    pattern: str = ""
    output_keys: List[str] = field(default_factory=list)
    default_output_key: str = "output"
    flags: int = 0


class RegexParser:
    """Parses LLM outputs using regex patterns."""
    
    def __init__(self, config: Optional[RegexParserConfig] = None):
        self.config = config or RegexParserConfig()
        self._compiled = re.compile(self.config.pattern, self.config.flags) if self.config.pattern else None
    
    def get_format_instructions(self) -> str:
        if self.config.output_keys:
            return f"Response should contain: {', '.join(self.config.output_keys)}"
        return ""
    
    def parse(self, text: str) -> Dict[str, Any]:
        if not self._compiled:
            return {self.config.default_output_key: text}
        match = self._compiled.search(text)
        if not match:
            raise ValueError(f"Pattern did not match: {self.config.pattern}")
        result = match.groupdict()
        if not result:
            groups = match.groups()
            if self.config.output_keys:
                result = dict(zip(self.config.output_keys, groups))
            else:
                result = {f"group_{i}": g for i, g in enumerate(groups)}
        return result
    
    def parse_all(self, text: str) -> List[Dict[str, Any]]:
        if not self._compiled:
            return [{self.config.default_output_key: text}]
        results = []
        for match in self._compiled.finditer(text):
            result = match.groupdict()
            if not result:
                groups = match.groups()
                result = dict(zip(self.config.output_keys, groups)) if self.config.output_keys else {f"group_{i}": g for i, g in enumerate(groups)}
            results.append(result)
        return results
    
    @classmethod
    def from_pattern(cls, pattern: str, output_keys: Optional[List[str]] = None, flags: int = 0) -> "RegexParser":
        return cls(RegexParserConfig(pattern=pattern, output_keys=output_keys or [], flags=flags))


def create_regex_parser(pattern: str, output_keys: Optional[List[str]] = None, **kwargs) -> RegexParser:
    return RegexParser(RegexParserConfig(pattern=pattern, output_keys=output_keys or [], **kwargs))


__all__ = ["RegexParser", "RegexParserConfig", "create_regex_parser"]
