"""
Web Search Tool
"""

import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SearchProvider(str, Enum):
    SERPAPI = "serpapi"
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"


@dataclass
class WebSearchConfig:
    provider: SearchProvider = SearchProvider.DUCKDUCKGO
    api_key: Optional[str] = None
    max_results: int = 5


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    position: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSearchTool:
    """Web search tool supporting multiple providers."""
    
    name = "web_search"
    description = "Search the web for information."
    
    def __init__(self, config: Optional[WebSearchConfig] = None):
        self.config = config or WebSearchConfig()
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if self.config.provider == SearchProvider.DUCKDUCKGO:
            return await self._search_duckduckgo(query, num_results)
        return await self._search_duckduckgo(query, num_results)
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
                position=i + 1
            ) for i, r in enumerate(results)]
        except ImportError:
            logger.error("duckduckgo_search not installed")
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []


__all__ = ["WebSearchTool", "WebSearchConfig", "SearchResult", "SearchProvider"]
