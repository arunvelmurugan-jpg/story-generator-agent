"""
MCP (Model Context Protocol) Module for PHTN.AI Sub-Agent Framework

Provides MCP client and server capabilities for tool interoperability
and standardized context sharing between AI agents.

Supports multiple providers:
- Azure (Azure AD authentication)
- AWS (IAM/SigV4 authentication)
- GCP (IAM authentication)
- Custom (Bearer, API Key, OAuth2, mTLS)
- Local (no auth)
- Anthropic, OpenAI (API key)

Reference: https://modelcontextprotocol.io/
"""

from .client import (
    MCPClient,
    MCPClientConfig,
    MCPTransport,
    MCPProvider,
    MCPAuthType,
    MCPAuthConfig,
)
from .server import MCPServer, MCPServerConfig
from .types import (
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPMessage,
    MCPToolCall,
    MCPToolResult,
    MCPServerInfo,
    MCPRequest,
    MCPResponse,
)
from .manager import MCPManager

__all__ = [
    "MCPClient",
    "MCPClientConfig",
    "MCPTransport",
    "MCPProvider",
    "MCPAuthType",
    "MCPAuthConfig",
    "MCPServer",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPMessage",
    "MCPToolCall",
    "MCPToolResult",
    "MCPServerInfo",
    "MCPRequest",
    "MCPResponse",
    "MCPManager",
]
