"""
Tool Registry for PHTN.AI Sub-Agent Framework

Manages registration and discovery of tools.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    """Registered tool information."""
    tool_id: str
    name: str
    description: str
    version: str
    type: str
    handler: Optional[Callable] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_llm_schema(self) -> Dict[str, Any]:
        """Convert to LLM tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "type": self.type,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "enabled": self.enabled,
        }


class ToolRegistry:
    """
    Registry for managing tools.
    
    Features:
    - Tool registration and discovery
    - Schema validation
    - Tool versioning
    - Enable/disable tools
    """
    
    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}
        self._handlers: Dict[str, Callable] = {}
        logger.debug("ToolRegistry initialized")
    
    def register(
        self,
        tool_config: Any,
        handler: Optional[Callable] = None,
    ) -> RegisteredTool:
        """
        Register a tool.
        
        Args:
            tool_config: Tool configuration (ToolDefinition or dict)
            handler: Optional tool handler function
            
        Returns:
            RegisteredTool
        """
        if hasattr(tool_config, "model_dump"):
            config = tool_config.model_dump()
        elif hasattr(tool_config, "__dict__"):
            config = vars(tool_config)
        else:
            config = dict(tool_config)
        
        tool = RegisteredTool(
            tool_id=config.get("tool_id", config.get("name")),
            name=config.get("name", config.get("tool_id")),
            description=config.get("description", ""),
            version=config.get("version", "1.0.0"),
            type=config.get("type", "BUILTIN"),
            handler=handler,
            input_schema=config.get("interface", {}).get("inputSchema", config.get("input_schema", {})),
            output_schema=config.get("interface", {}).get("outputSchema", config.get("output_schema", {})),
            config=config.get("config", {}),
            enabled=config.get("enabled", True),
        )
        
        self._tools[tool.tool_id] = tool
        
        if handler:
            self._handlers[tool.tool_id] = handler
        
        logger.info(f"Tool registered: {tool.tool_id} ({tool.type})")
        return tool
    
    def register_function(
        self,
        tool_id: str,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> RegisteredTool:
        """
        Register a function as a tool.
        
        Args:
            tool_id: Tool identifier
            name: Tool name
            description: Tool description
            handler: Function to execute
            input_schema: Input JSON schema
            
        Returns:
            RegisteredTool
        """
        tool = RegisteredTool(
            tool_id=tool_id,
            name=name,
            description=description,
            version="1.0.0",
            type="FUNCTION",
            handler=handler,
            input_schema=input_schema or {"type": "object", "properties": {}},
        )
        
        self._tools[tool_id] = tool
        self._handlers[tool_id] = handler
        
        logger.info(f"Function tool registered: {tool_id}")
        return tool
    
    def get(self, tool_id: str) -> Optional[RegisteredTool]:
        """Get tool by ID."""
        return self._tools.get(tool_id)
    
    def get_handler(self, tool_id: str) -> Optional[Callable]:
        """Get tool handler by ID."""
        return self._handlers.get(tool_id)
    
    def list_tools(self, enabled_only: bool = True) -> List[RegisteredTool]:
        """List all registered tools."""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools
    
    def get_tool_schemas(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM."""
        tools = self.list_tools(enabled_only)
        return [t.to_llm_schema() for t in tools]
    
    def enable(self, tool_id: str) -> bool:
        """Enable a tool."""
        tool = self._tools.get(tool_id)
        if tool:
            tool.enabled = True
            return True
        return False
    
    def disable(self, tool_id: str) -> bool:
        """Disable a tool."""
        tool = self._tools.get(tool_id)
        if tool:
            tool.enabled = False
            return True
        return False
    
    def unregister(self, tool_id: str) -> bool:
        """Unregister a tool."""
        if tool_id in self._tools:
            del self._tools[tool_id]
            self._handlers.pop(tool_id, None)
            logger.info(f"Tool unregistered: {tool_id}")
            return True
        return False
    
    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._handlers.clear()
