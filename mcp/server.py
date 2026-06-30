"""
MCP Server for PHTN.AI Sub-Agent Framework

Implements Model Context Protocol server to expose agent tools,
resources, and prompts to other MCP clients.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from ..observability.otel_logging import get_logger
from .types import (
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPToolParameter,
    MCPRequest,
    MCPResponse,
    MCPServerInfo,
)

logger = get_logger(__name__)


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    name: str = "phtnai-subagent"
    version: str = "1.0.0"
    protocol_version: str = "1.0"
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "tools": True,
        "resources": True,
        "prompts": True,
        "streaming": False,
    })


ToolHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


class MCPServer:
    """
    MCP Server for exposing agent capabilities.
    
    Exposes:
    - Agent tools as MCP tools
    - Knowledge base as MCP resources
    - Prompt templates as MCP prompts
    """
    
    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self._tools: Dict[str, MCPTool] = {}
        self._tool_handlers: Dict[str, ToolHandler] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._initialized = False
        
        logger.info(f"MCP Server initialized: {config.name}")
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: List[MCPToolParameter],
        handler: ToolHandler,
    ):
        """
        Register a tool with the MCP server.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
            handler: Async function to handle tool calls
        """
        tool = MCPTool(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._tools[name] = tool
        self._tool_handlers[name] = handler
        logger.debug(f"Registered MCP tool: {name}")
    
    def register_resource(self, resource: MCPResource):
        """
        Register a resource with the MCP server.
        
        Args:
            resource: Resource to register
        """
        self._resources[resource.uri] = resource
        logger.debug(f"Registered MCP resource: {resource.uri}")
    
    def register_prompt(self, prompt: MCPPrompt):
        """
        Register a prompt template with the MCP server.
        
        Args:
            prompt: Prompt template to register
        """
        self._prompts[prompt.name] = prompt
        logger.debug(f"Registered MCP prompt: {prompt.name}")
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP JSON-RPC request.
        
        Args:
            request_data: JSON-RPC request
            
        Returns:
            JSON-RPC response
        """
        method = request_data.get("method", "")
        params = request_data.get("params", {})
        request_id = request_data.get("id")
        
        handlers = {
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "shutdown": self._handle_shutdown,
        }
        
        handler = handlers.get(method)
        if not handler:
            return MCPResponse(
                id=request_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            ).to_dict()
        
        try:
            result = await handler(params)
            return MCPResponse(
                id=request_id,
                result=result,
            ).to_dict()
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return MCPResponse(
                id=request_id,
                error={
                    "code": -32603,
                    "message": str(e),
                },
            ).to_dict()
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": self.config.protocol_version,
            "capabilities": self.config.capabilities,
            "serverInfo": {
                "name": self.config.name,
                "version": self.config.version,
            },
        }
    
    async def _handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification."""
        logger.info("MCP client initialized")
        return None
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [tool.to_dict() for tool in self._tools.values()],
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name not in self._tool_handlers:
            raise ValueError(f"Tool not found: {name}")
        
        handler = self._tool_handlers[name]
        result = await handler(arguments)
        
        if isinstance(result, str):
            content = [{"type": "text", "text": result}]
        elif isinstance(result, dict):
            content = [{"type": "text", "text": json.dumps(result)}]
        else:
            content = [{"type": "text", "text": str(result)}]
        
        return {"content": content}
    
    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {
            "resources": [res.to_dict() for res in self._resources.values()],
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if uri not in self._resources:
            raise ValueError(f"Resource not found: {uri}")
        
        resource = self._resources[uri]
        
        content = resource.content
        if content is None:
            content = ""
        
        if isinstance(content, bytes):
            import base64
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "blob": base64.b64encode(content).decode(),
                }],
            }
        else:
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": str(content),
                }],
            }
    
    async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        return {
            "prompts": [prompt.to_dict() for prompt in self._prompts.values()],
        }
    
    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name not in self._prompts:
            raise ValueError(f"Prompt not found: {name}")
        
        prompt = self._prompts[name]
        rendered = prompt.render(**arguments)
        
        return {
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": rendered},
            }],
        }
    
    async def _handle_shutdown(self, params: Dict[str, Any]) -> None:
        """Handle shutdown request."""
        logger.info("MCP server shutdown requested")
        self._initialized = False
        return None
    
    def create_router(self) -> APIRouter:
        """
        Create FastAPI router for MCP endpoints.
        
        Returns:
            FastAPI router
        """
        router = APIRouter(prefix="/mcp", tags=["MCP"])
        
        @router.post("/")
        async def mcp_endpoint(request: Request):
            """MCP JSON-RPC endpoint."""
            try:
                data = await request.json()
                response = await self.handle_request(data)
                return JSONResponse(content=response)
            except Exception as e:
                logger.error(f"MCP endpoint error: {e}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": str(e)},
                    },
                    status_code=400,
                )
        
        @router.get("/info")
        async def mcp_info():
            """Get MCP server information."""
            return {
                "name": self.config.name,
                "version": self.config.version,
                "protocolVersion": self.config.protocol_version,
                "capabilities": self.config.capabilities,
                "tools": len(self._tools),
                "resources": len(self._resources),
                "prompts": len(self._prompts),
            }
        
        return router
    
    def get_server_info(self) -> MCPServerInfo:
        """Get server information."""
        return MCPServerInfo(
            name=self.config.name,
            version=self.config.version,
            uri="/mcp",
            protocol_version=self.config.protocol_version,
            capabilities=self.config.capabilities,
            tools=list(self._tools.values()),
            resources=list(self._resources.values()),
            prompts=list(self._prompts.values()),
        )
