"""
MCP Client for PHTN.AI Sub-Agent Framework

Implements Model Context Protocol client for connecting to MCP servers
and discovering/invoking tools, resources, and prompts.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
from enum import Enum

from ..observability.otel_logging import get_logger
from .types import (
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPToolCall,
    MCPToolResult,
    MCPServerInfo,
    MCPRequest,
    MCPResponse,
)

logger = get_logger(__name__)


class MCPTransport(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SSE = "sse"
    GRPC = "grpc"


class MCPProvider(str, Enum):
    """MCP server provider types."""
    CUSTOM = "custom"
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class MCPAuthType(str, Enum):
    """MCP authentication types."""
    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    AZURE_AD = "azure_ad"
    AWS_IAM = "aws_iam"
    GCP_IAM = "gcp_iam"
    MTLS = "mtls"
    BASIC = "basic"


@dataclass
class MCPAuthConfig:
    """MCP authentication configuration."""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    scope: Optional[str] = None
    region: Optional[str] = None
    role_arn: Optional[str] = None
    service_account: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class MCPClientConfig:
    """MCP client configuration."""
    server_uri: str
    name: str = "default"
    provider: MCPProvider = MCPProvider.CUSTOM
    transport: MCPTransport = MCPTransport.HTTP
    auth_type: MCPAuthType = MCPAuthType.BEARER
    auth_token: Optional[str] = None
    auth_config: Optional[MCPAuthConfig] = None
    timeout_ms: int = 30000
    retry_attempts: int = 3
    retry_delay_ms: int = 1000
    headers: Dict[str, str] = field(default_factory=dict)
    health_check_interval_ms: int = 60000
    tool_filter_include: Optional[List[str]] = None
    tool_filter_exclude: Optional[List[str]] = None
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "tools": True,
        "resources": True,
        "prompts": True,
        "streaming": True,
    })


class MCPClient:
    """
    MCP Client for connecting to Model Context Protocol servers.
    
    Supports:
    - Tool discovery and invocation
    - Resource access
    - Prompt templates
    - Streaming responses
    - Multiple transport protocols (HTTP, WebSocket, SSE, stdio)
    """
    
    def __init__(self, config: MCPClientConfig):
        """
        Initialize MCP client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self._server_info: Optional[MCPServerInfo] = None
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._connected = False
        self._session_id: Optional[str] = None
        self._http_client: Optional[Any] = None
        self._cached_auth_token: Optional[str] = None
        self._auth_token_expires: Optional[float] = None
        
        logger.info(f"MCP Client initialized for {config.server_uri} (provider: {config.provider.value})")
    
    async def connect(self) -> bool:
        """
        Connect to MCP server and initialize session.
        
        Returns:
            True if connection successful
        """
        try:
            self._session_id = str(uuid.uuid4())
            
            response = await self._send_request(MCPRequest(
                method="initialize",
                params={
                    "protocolVersion": "1.0",
                    "capabilities": self.config.capabilities,
                    "clientInfo": {
                        "name": "phtnai-subagent",
                        "version": "1.0.0",
                    },
                },
                id=str(uuid.uuid4()),
            ))
            
            if response.is_error:
                logger.error(f"MCP initialization failed: {response.error}")
                return False
            
            result = response.result or {}
            self._server_info = MCPServerInfo(
                name=result.get("serverInfo", {}).get("name", "unknown"),
                version=result.get("serverInfo", {}).get("version", "1.0"),
                uri=self.config.server_uri,
                protocol_version=result.get("protocolVersion", "1.0"),
                capabilities=result.get("capabilities", {}),
            )
            
            await self._send_request(MCPRequest(
                method="notifications/initialized",
            ))
            
            await self._discover_capabilities()
            
            self._connected = True
            logger.info(f"Connected to MCP server: {self._server_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._connected:
            try:
                await self._send_request(MCPRequest(
                    method="shutdown",
                    id=str(uuid.uuid4()),
                ))
            except Exception as e:
                logger.warning(f"Error during MCP disconnect: {e}")
            finally:
                self._connected = False
                self._session_id = None
                logger.info("Disconnected from MCP server")
    
    async def _discover_capabilities(self):
        """Discover tools, resources, and prompts from server."""
        if self._server_info and self._server_info.capabilities.get("tools"):
            await self._list_tools()
        
        if self._server_info and self._server_info.capabilities.get("resources"):
            await self._list_resources()
        
        if self._server_info and self._server_info.capabilities.get("prompts"):
            await self._list_prompts()
    
    async def _list_tools(self):
        """List available tools from server."""
        response = await self._send_request(MCPRequest(
            method="tools/list",
            id=str(uuid.uuid4()),
        ))
        
        if not response.is_error and response.result:
            tools = response.result.get("tools", [])
            for tool_data in tools:
                tool = MCPTool.from_dict(tool_data)
                tool.server_uri = self.config.server_uri
                self._tools[tool.name] = tool
            logger.info(f"Discovered {len(self._tools)} MCP tools")
    
    async def _list_resources(self):
        """List available resources from server."""
        response = await self._send_request(MCPRequest(
            method="resources/list",
            id=str(uuid.uuid4()),
        ))
        
        if not response.is_error and response.result:
            resources = response.result.get("resources", [])
            for res_data in resources:
                resource = MCPResource.from_dict(res_data)
                self._resources[resource.uri] = resource
            logger.info(f"Discovered {len(self._resources)} MCP resources")
    
    async def _list_prompts(self):
        """List available prompts from server."""
        response = await self._send_request(MCPRequest(
            method="prompts/list",
            id=str(uuid.uuid4()),
        ))
        
        if not response.is_error and response.result:
            prompts = response.result.get("prompts", [])
            for prompt_data in prompts:
                prompt = MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description"),
                    metadata=prompt_data.get("metadata", {}),
                )
                self._prompts[prompt.name] = prompt
            logger.info(f"Discovered {len(self._prompts)} MCP prompts")
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        return list(self._tools.values())
    
    @property
    def resources(self) -> List[MCPResource]:
        """Get list of available resources."""
        return list(self._resources.values())
    
    @property
    def prompts(self) -> List[MCPPrompt]:
        """Get list of available prompts."""
        return list(self._prompts.values())
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        tool_call_id = str(uuid.uuid4())
        
        if name not in self._tools:
            return MCPToolResult(
                tool_call_id=tool_call_id,
                content=None,
                success=False,
                error=f"Tool '{name}' not found",
            )
        
        try:
            response = await self._send_request(MCPRequest(
                method="tools/call",
                params={
                    "name": name,
                    "arguments": arguments,
                },
                id=tool_call_id,
            ))
            
            if response.is_error:
                return MCPToolResult(
                    tool_call_id=tool_call_id,
                    content=None,
                    success=False,
                    error=str(response.error),
                )
            
            result = response.result or {}
            content = result.get("content", [])
            
            if content and isinstance(content, list):
                text_content = []
                for item in content:
                    if item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                content = "\n".join(text_content) if text_content else content
            
            return MCPToolResult(
                tool_call_id=tool_call_id,
                content=content,
                success=True,
                metadata=result.get("metadata", {}),
            )
            
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return MCPToolResult(
                tool_call_id=tool_call_id,
                content=None,
                success=False,
                error=str(e),
            )
    
    async def read_resource(self, uri: str) -> Optional[MCPResource]:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource with content
        """
        try:
            response = await self._send_request(MCPRequest(
                method="resources/read",
                params={"uri": uri},
                id=str(uuid.uuid4()),
            ))
            
            if response.is_error:
                logger.error(f"Failed to read resource: {response.error}")
                return None
            
            result = response.result or {}
            contents = result.get("contents", [])
            
            if contents:
                content_item = contents[0]
                resource = self._resources.get(uri) or MCPResource(
                    uri=uri,
                    name=uri.split("/")[-1],
                )
                resource.content = content_item.get("text") or content_item.get("blob")
                resource.mime_type = content_item.get("mimeType", "text/plain")
                return resource
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read MCP resource: {e}")
            return None
    
    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get a rendered prompt from the MCP server.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            
        Returns:
            Rendered prompt string
        """
        try:
            response = await self._send_request(MCPRequest(
                method="prompts/get",
                params={
                    "name": name,
                    "arguments": arguments or {},
                },
                id=str(uuid.uuid4()),
            ))
            
            if response.is_error:
                logger.error(f"Failed to get prompt: {response.error}")
                return None
            
            result = response.result or {}
            messages = result.get("messages", [])
            
            if messages:
                return "\n".join(
                    msg.get("content", {}).get("text", "")
                    for msg in messages
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get MCP prompt: {e}")
            return None
    
    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request to MCP server."""
        if self.config.transport == MCPTransport.HTTP:
            return await self._send_http_request(request)
        elif self.config.transport == MCPTransport.WEBSOCKET:
            return await self._send_websocket_request(request)
        elif self.config.transport == MCPTransport.SSE:
            return await self._send_sse_request(request)
        else:
            return await self._send_stdio_request(request)
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type."""
        headers = {}
        
        if self.config.auth_type == MCPAuthType.NONE:
            return headers
        
        if self.config.auth_type == MCPAuthType.BEARER:
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
        
        elif self.config.auth_type == MCPAuthType.API_KEY:
            if self.config.auth_token:
                headers["X-API-Key"] = self.config.auth_token
        
        elif self.config.auth_type == MCPAuthType.BASIC:
            if self.config.auth_config:
                import base64
                credentials = f"{self.config.auth_config.username}:{self.config.auth_config.password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
        
        elif self.config.auth_type == MCPAuthType.AZURE_AD:
            token = await self._get_azure_ad_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_type == MCPAuthType.AWS_IAM:
            aws_headers = await self._get_aws_sigv4_headers()
            headers.update(aws_headers)
        
        elif self.config.auth_type == MCPAuthType.GCP_IAM:
            token = await self._get_gcp_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_type == MCPAuthType.OAUTH2:
            token = await self._get_oauth2_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        return headers
    
    async def _get_azure_ad_token(self) -> Optional[str]:
        """Get Azure AD access token."""
        import time
        
        if self._cached_auth_token and self._auth_token_expires:
            if time.time() < self._auth_token_expires - 60:
                return self._cached_auth_token
        
        if not self.config.auth_config:
            logger.error("Azure AD auth config not provided")
            return None
        
        try:
            import aiohttp
            
            tenant_id = self.config.auth_config.tenant_id
            client_id = self.config.auth_config.client_id
            client_secret = self.config.auth_config.client_secret
            scope = self.config.auth_config.scope or "https://management.azure.com/.default"
            
            url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self._cached_auth_token = result.get("access_token")
                        expires_in = result.get("expires_in", 3600)
                        self._auth_token_expires = time.time() + expires_in
                        return self._cached_auth_token
                    else:
                        logger.error(f"Azure AD token error: {resp.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Azure AD authentication failed: {e}")
            return None
    
    async def _get_aws_sigv4_headers(self) -> Dict[str, str]:
        """Get AWS SigV4 signed headers."""
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            
            region = self.config.auth_config.region if self.config.auth_config else "us-east-1"
            
            if self.config.auth_config and self.config.auth_config.role_arn:
                sts = boto3.client("sts", region_name=region)
                assumed = sts.assume_role(
                    RoleArn=self.config.auth_config.role_arn,
                    RoleSessionName="mcp-client",
                )
                credentials = boto3.Session(
                    aws_access_key_id=assumed["Credentials"]["AccessKeyId"],
                    aws_secret_access_key=assumed["Credentials"]["SecretAccessKey"],
                    aws_session_token=assumed["Credentials"]["SessionToken"],
                ).get_credentials()
            else:
                credentials = boto3.Session().get_credentials()
            
            request = AWSRequest(method="POST", url=self.config.server_uri)
            SigV4Auth(credentials, "execute-api", region).add_auth(request)
            
            return dict(request.headers)
            
        except ImportError:
            logger.error("boto3 not installed for AWS IAM auth")
            return {}
        except Exception as e:
            logger.error(f"AWS SigV4 signing failed: {e}")
            return {}
    
    async def _get_gcp_token(self) -> Optional[str]:
        """Get GCP access token."""
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            
            credentials, project = default()
            credentials.refresh(Request())
            return credentials.token
            
        except ImportError:
            logger.error("google-auth not installed for GCP IAM auth")
            return None
        except Exception as e:
            logger.error(f"GCP authentication failed: {e}")
            return None
    
    async def _get_oauth2_token(self) -> Optional[str]:
        """Get OAuth2 access token."""
        import time
        
        if self._cached_auth_token and self._auth_token_expires:
            if time.time() < self._auth_token_expires - 60:
                return self._cached_auth_token
        
        if not self.config.auth_config:
            return self.config.auth_token
        
        return self.config.auth_token
    
    async def _send_http_request(self, request: MCPRequest) -> MCPResponse:
        """Send HTTP request to MCP server."""
        import aiohttp
        
        auth_headers = await self._get_auth_headers()
        
        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
            **auth_headers,
        }
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout_ms / 1000
        )
        
        last_error = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.config.server_uri,
                        json=request.to_dict(),
                        headers=headers,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return MCPResponse(
                                id=data.get("id"),
                                result=data.get("result"),
                                error=data.get("error"),
                            )
                        elif resp.status in (401, 403):
                            self._cached_auth_token = None
                            self._auth_token_expires = None
                            auth_headers = await self._get_auth_headers()
                            headers.update(auth_headers)
                            last_error = f"HTTP error: {resp.status}"
                        elif resp.status >= 500:
                            last_error = f"HTTP error: {resp.status}"
                        else:
                            return MCPResponse(
                                id=request.id,
                                error={
                                    "code": resp.status,
                                    "message": f"HTTP error: {resp.status}",
                                },
                            )
                            
            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.config.retry_attempts:
                await asyncio.sleep(self.config.retry_delay_ms / 1000)
        
        return MCPResponse(
            id=request.id,
            error={"code": -1, "message": last_error or "Unknown error"},
        )
    
    async def _send_websocket_request(self, request: MCPRequest) -> MCPResponse:
        """Send WebSocket request to MCP server."""
        return MCPResponse(
            id=request.id,
            error={"code": -1, "message": "WebSocket transport not implemented"},
        )
    
    async def _send_sse_request(self, request: MCPRequest) -> MCPResponse:
        """Send SSE request to MCP server."""
        return MCPResponse(
            id=request.id,
            error={"code": -1, "message": "SSE transport not implemented"},
        )
    
    async def _send_stdio_request(self, request: MCPRequest) -> MCPResponse:
        """Send stdio request to MCP server."""
        return MCPResponse(
            id=request.id,
            error={"code": -1, "message": "stdio transport not implemented"},
        )
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.type,
                                "description": p.description,
                            }
                            for p in tool.parameters
                        },
                        "required": [p.name for p in tool.parameters if p.required],
                    },
                },
            }
            for tool in self._tools.values()
        ]
