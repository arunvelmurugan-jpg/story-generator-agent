"""
MCP Types for PHTN.AI Sub-Agent Framework

Defines data types for Model Context Protocol communication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime


class MCPMessageRole(str, Enum):
    """MCP message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MCPToolType(str, Enum):
    """MCP tool types."""
    FUNCTION = "function"
    HTTP = "http"
    DATABASE = "database"
    FILE = "file"
    CUSTOM = "custom"


class MCPResourceType(str, Enum):
    """MCP resource types."""
    TEXT = "text"
    BLOB = "blob"
    FILE = "file"
    URI = "uri"
    TEMPLATE = "template"


@dataclass
class MCPToolParameter:
    """MCP tool parameter definition."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
        }
        if self.description:
            result["description"] = self.description
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        return result


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    tool_type: MCPToolType = MCPToolType.FUNCTION
    server_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
            "toolType": self.tool_type.value,
            "serverUri": self.server_uri,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        parameters = []
        if "parameters" in data:
            props = data["parameters"].get("properties", {})
            required = data["parameters"].get("required", [])
            for name, prop in props.items():
                parameters.append(MCPToolParameter(
                    name=name,
                    type=prop.get("type", "string"),
                    description=prop.get("description"),
                    required=name in required,
                    default=prop.get("default"),
                    enum=prop.get("enum"),
                ))
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=parameters,
            tool_type=MCPToolType(data.get("toolType", "function")),
            server_uri=data.get("serverUri"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: str = "text/plain"
    resource_type: MCPResourceType = MCPResourceType.TEXT
    content: Optional[Union[str, bytes]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
            "resourceType": self.resource_type.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        return cls(
            uri=data["uri"],
            name=data["name"],
            description=data.get("description"),
            mime_type=data.get("mimeType", "text/plain"),
            resource_type=MCPResourceType(data.get("resourceType", "text")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPPrompt:
    """MCP prompt template."""
    name: str
    description: Optional[str] = None
    arguments: List[MCPToolParameter] = field(default_factory=list)
    template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [a.to_dict() for a in self.arguments],
            "metadata": self.metadata,
        }
    
    def render(self, **kwargs) -> str:
        """Render the prompt template with arguments."""
        if not self.template:
            return ""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


@dataclass
class MCPMessage:
    """MCP message."""
    role: MCPMessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List["MCPToolCall"]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class MCPToolCall:
    """MCP tool call request."""
    id: str
    name: str
    arguments: Dict[str, Any]
    server_uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
            "serverUri": self.server_uri,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPToolCall":
        func = data.get("function", {})
        return cls(
            id=data["id"],
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
            server_uri=data.get("serverUri"),
        )


@dataclass
class MCPToolResult:
    """MCP tool call result."""
    tool_call_id: str
    content: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class MCPServerInfo:
    """MCP server information."""
    name: str
    version: str
    uri: str
    protocol_version: str = "1.0"
    capabilities: Dict[str, bool] = field(default_factory=dict)
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)
    prompts: List[MCPPrompt] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "uri": self.uri,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "tools": [t.to_dict() for t in self.tools],
            "resources": [r.to_dict() for r in self.resources],
            "prompts": [p.to_dict() for p in self.prompts],
        }


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params:
            result["params"] = self.params
        if self.id:
            result["id"] = self.id
        return result


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
        }
        if self.id:
            result["id"] = self.id
        if self.result is not None:
            result["result"] = self.result
        if self.error:
            result["error"] = self.error
        return result
    
    @property
    def is_error(self) -> bool:
        return self.error is not None
