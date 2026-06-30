"""
Wikipedia Tool
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WikipediaResult:
    title: str
    summary: str
    url: str
    content: Optional[str] = None


class WikipediaTool:
    """Wikipedia tool for searching and retrieving articles."""
    
    name = "wikipedia"
    description = "Search Wikipedia for information."
    
    def __init__(self, language: str = "en"):
        self.language = language
        self._wiki = None
    
    def _get_wiki(self):
        if self._wiki is None:
            try:
                import wikipedia
                wikipedia.set_lang(self.language)
                self._wiki = wikipedia
            except ImportError:
                raise ImportError("wikipedia required: pip install wikipedia")
        return self._wiki
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "sentences": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, sentences: int = 3) -> WikipediaResult:
        wiki = self._get_wiki()
        try:
            page = wiki.page(query, auto_suggest=True)
            return WikipediaResult(
                title=page.title,
                summary=wiki.summary(query, sentences=sentences),
                url=page.url
            )
        except wiki.exceptions.DisambiguationError as e:
            return WikipediaResult(
                title=query,
                summary=f"Multiple results: {', '.join(e.options[:5])}",
                url=""
            )
        except wiki.exceptions.PageError:
            return WikipediaResult(title=query, summary=f"No article found for '{query}'", url="")
        except Exception as e:
            return WikipediaResult(title=query, summary=f"Error: {e}", url="")


__all__ = ["WikipediaTool", "WikipediaResult"]
