"""
MCP Manager for PHTN.AI Sub-Agent Framework

Manages MCP clients and server, providing unified access to
MCP tools across multiple servers.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..observability.otel_logging import get_logger
from .client import MCPClient, MCPClientConfig, MCPTransport
from .server import MCPServer, MCPServerConfig
from .types import MCPTool, MCPToolResult, MCPResource, MCPToolParameter

if TYPE_CHECKING:
    from ..core.config_loader import MCPConfig

logger = get_logger(__name__)


@dataclass
class MCPServerConnection:
    """MCP server connection configuration."""
    uri: str
    name: str
    transport: str = "http"
    auth_token: Optional[str] = None
    enabled: bool = True
    timeout_ms: int = 30000


class MCPManager:
    """
    Manages MCP client connections and server.
    
    Features:
    - Connect to multiple MCP servers
    - Aggregate tools from all servers
    - Route tool calls to appropriate server
    - Expose local tools via MCP server
    """
    
    def __init__(self, config: Optional["MCPConfig"] = None):
        """
        Initialize MCP manager.
        
        Args:
            config: MCP configuration from PHTN-AGENT.json
        """
        self.config = config
        self._clients: Dict[str, MCPClient] = {}
        self._server: Optional[MCPServer] = None
        self._all_tools: Dict[str, MCPTool] = {}
        self._tool_to_server: Dict[str, str] = {}
        self._initialized = False
        
        logger.info("MCP Manager initialized")
    
    async def initialize(self):
        """Initialize MCP manager with configured servers."""
        if self._initialized:
            return
        
        if not self.config or not self.config.enabled:
            logger.info("MCP is disabled in configuration")
            return
        
        if hasattr(self.config, 'servers') and self.config.servers:
            for server_config in self.config.servers:
                await self.connect_to_server(MCPServerConnection(
                    uri=server_config.get("uri", ""),
                    name=server_config.get("name", "unknown"),
                    transport=server_config.get("transport", "http"),
                    auth_token=server_config.get("auth_token"),
                    enabled=server_config.get("enabled", True),
                    timeout_ms=server_config.get("timeout_ms", 30000),
                ))
        
        if self.config.capabilities.get("tools", True):
            await self._setup_server()
        
        self._initialized = True
        logger.info(f"MCP Manager initialized with {len(self._clients)} clients")
    
    async def connect_to_server(self, connection: MCPServerConnection) -> bool:
        """
        Connect to an MCP server.
        
        Args:
            connection: Server connection configuration
            
        Returns:
            True if connection successful
        """
        if not connection.enabled:
            logger.info(f"MCP server {connection.name} is disabled")
            return False
        
        if connection.name in self._clients:
            logger.warning(f"MCP server {connection.name} already connected")
            return True
        
        try:
            transport = MCPTransport(connection.transport)
        except ValueError:
            transport = MCPTransport.HTTP
        
        client_config = MCPClientConfig(
            server_uri=connection.uri,
            transport=transport,
            timeout_ms=connection.timeout_ms,
            auth_token=connection.auth_token,
        )
        
        client = MCPClient(client_config)
        
        if await client.connect():
            self._clients[connection.name] = client
            
            for tool in client.tools:
                self._all_tools[tool.name] = tool
                self._tool_to_server[tool.name] = connection.name
            
            logger.info(f"Connected to MCP server: {connection.name} ({len(client.tools)} tools)")
            return True
        else:
            logger.error(f"Failed to connect to MCP server: {connection.name}")
            return False
    
    async def disconnect_from_server(self, name: str):
        """
        Disconnect from an MCP server.
        
        Args:
            name: Server name
        """
        if name not in self._clients:
            return
        
        client = self._clients[name]
        await client.disconnect()
        
        tools_to_remove = [
            tool_name for tool_name, server_name in self._tool_to_server.items()
            if server_name == name
        ]
        for tool_name in tools_to_remove:
            del self._all_tools[tool_name]
            del self._tool_to_server[tool_name]
        
        del self._clients[name]
        logger.info(f"Disconnected from MCP server: {name}")
    
    async def _setup_server(self):
        """Set up local MCP server."""
        server_config = MCPServerConfig(
            name=self.config.server_name if hasattr(self.config, 'server_name') else "phtnai-subagent",
            version="1.0.0",
            capabilities=self.config.capabilities if hasattr(self.config, 'capabilities') else {},
        )
        self._server = MCPServer(server_config)
        logger.info("MCP server initialized")
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get all available MCP tools."""
        return list(self._all_tools.values())
    
    @property
    def server(self) -> Optional[MCPServer]:
        """Get local MCP server."""
        return self._server
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        return self._all_tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._all_tools
    
    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> MCPToolResult:
        """
        Call an MCP tool.
        
        Routes the call to the appropriate MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if name not in self._tool_to_server:
            return MCPToolResult(
                tool_call_id="",
                content=None,
                success=False,
                error=f"MCP tool '{name}' not found",
            )
        
        server_name = self._tool_to_server[name]
        client = self._clients.get(server_name)
        
        if not client:
            return MCPToolResult(
                tool_call_id="",
                content=None,
                success=False,
                error=f"MCP server '{server_name}' not connected",
            )
        
        return await client.call_tool(name, arguments)
    
    async def read_resource(self, uri: str) -> Optional[MCPResource]:
        """
        Read a resource from MCP servers.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource with content
        """
        for client in self._clients.values():
            for resource in client.resources:
                if resource.uri == uri:
                    return await client.read_resource(uri)
        return None
    
    def register_local_tool(
        self,
        name: str,
        description: str,
        parameters: List[MCPToolParameter],
        handler,
    ):
        """
        Register a local tool to expose via MCP server.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
            handler: Async handler function
        """
        if self._server:
            self._server.register_tool(name, description, parameters, handler)
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all MCP tools to OpenAI function calling format."""
        tools = []
        for client in self._clients.values():
            tools.extend(client.to_openai_tools())
        return tools
    
    async def close(self):
        """Close all MCP connections."""
        for name in list(self._clients.keys()):
            await self.disconnect_from_server(name)
        self._initialized = False
        logger.info("MCP Manager closed")
