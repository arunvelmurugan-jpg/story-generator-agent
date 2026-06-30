"""
Built-in Tools Module - n8n Compatible

Provides commonly used tools out of the box:
- Calculator: Mathematical operations
- WebSearch: Search the web (SerpAPI, Google, Bing)
- Wikipedia: Wikipedia lookups
- ThinkTool: Agent reasoning/reflection
- CodeExecutor: Execute Python code
- HttpRequest: Make HTTP requests
- FileSystem: File operations
- DateTime: Date/time operations

Aligned with n8n's built-in tool nodes.
"""

from .calculator import CalculatorTool, calculate
from .web_search import WebSearchTool, WebSearchConfig
from .wikipedia import WikipediaTool
from .think import ThinkTool, ThinkResult
from .code_executor import CodeExecutorTool, CodeExecutorConfig
from .http_request import HttpRequestTool, HttpRequestConfig
from .datetime_tool import DateTimeTool

__all__ = [
    "CalculatorTool",
    "calculate",
    "WebSearchTool",
    "WebSearchConfig",
    "WikipediaTool",
    "ThinkTool",
    "ThinkResult",
    "CodeExecutorTool",
    "CodeExecutorConfig",
    "HttpRequestTool",
    "HttpRequestConfig",
    "DateTimeTool",
]
