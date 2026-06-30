"""
Tool Executor for PHTN.AI Sub-Agent Framework

Executes tools with sandboxing, timeout, and error handling.
Now integrates with GuardrailsEngine for tool allowlist/denylist enforcement.
"""

import asyncio
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from .registry import ToolRegistry, RegisteredTool
from ..observability.otel_logging import get_logger, log_tool_call

if TYPE_CHECKING:
    from ..guardrails.engine import GuardrailsEngine

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools with safety features.
    
    Features:
    - Guardrails integration (allowlist/denylist enforcement)
    - Timeout enforcement
    - Input validation
    - Sandboxing support
    - Error handling
    - Execution logging
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        sandboxing_config: Optional[Dict[str, Any]] = None,
        default_timeout_ms: int = 30000,
        guardrails_engine: Optional["GuardrailsEngine"] = None,
    ):
        """
        Initialize ToolExecutor.
        
        Args:
            registry: Tool registry
            sandboxing_config: Sandboxing configuration
            default_timeout_ms: Default timeout in milliseconds
            guardrails_engine: Guardrails engine for tool validation
        """
        self.registry = registry
        self.sandboxing_config = sandboxing_config or {}
        self.default_timeout_ms = default_timeout_ms
        self.guardrails_engine = guardrails_engine
        
        logger.debug("ToolExecutor initialized")
    
    async def execute(
        self,
        tool_id: str,
        input_data: Dict[str, Any],
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool with guardrails enforcement.
        
        Args:
            tool_id: Tool identifier
            input_data: Tool input parameters
            timeout_ms: Execution timeout
            
        Returns:
            Execution result dictionary
        """
        if self.guardrails_engine:
            guardrail_result = await self.guardrails_engine.check_tool_call(tool_id, input_data)
            if not guardrail_result.passed:
                logger.warning(f"🚫 Tool '{tool_id}' blocked by guardrails: {guardrail_result.reason}")
                return {
                    "success": False,
                    "error": f"Tool blocked by guardrails: {guardrail_result.reason}",
                    "guardrail_result": guardrail_result.to_dict(),
                }
        
        tool = self.registry.get(tool_id)
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool not found: {tool_id}",
            }
        
        if not tool.enabled:
            return {
                "success": False,
                "error": f"Tool is disabled: {tool_id}",
            }
        
        logger.info(f"🔧 Executing tool: {tool_id}")
        
        handler = self.registry.get_handler(tool_id)
        
        if not handler:
            if tool.type == "HTTP":
                return await self._execute_http_tool(tool, input_data)
            elif tool.type == "MCP":
                return await self._execute_mcp_tool(tool, input_data)
            else:
                return {
                    "success": False,
                    "error": f"No handler registered for tool: {tool_id}",
                }
        
        timeout = (timeout_ms or self.default_timeout_ms) / 1000
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**input_data),
                    timeout=timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: handler(**input_data)
                    ),
                    timeout=timeout,
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log tool call for phtnai-ops-metrics TechOps tracking
            log_tool_call(
                logger,
                f"Tool executed successfully: {tool_id}",
                tool_name=tool.name,
                tool_id=tool_id,
                latency_ms=execution_time,
                status=200,
                success=True,
            )
            
            return {
                "success": True,
                "result": result,
                "tool_id": tool_id,
                "execution_time_ms": execution_time,
            }
            
        except asyncio.TimeoutError:
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log tool timeout for phtnai-ops-metrics TechOps tracking
            log_tool_call(
                logger,
                f"Tool execution timed out: {tool_id}",
                tool_name=tool.name,
                tool_id=tool_id,
                latency_ms=execution_time,
                status=504,
                success=False,
                error="timeout",
            )
            
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout}s",
                "tool_id": tool_id,
            }
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Tool execution failed: {tool_id} - {e}")
            
            # Log tool failure for phtnai-ops-metrics TechOps tracking
            log_tool_call(
                logger,
                f"Tool execution failed: {tool_id}",
                tool_name=tool.name,
                tool_id=tool_id,
                latency_ms=execution_time,
                status=500,
                success=False,
                error=str(e),
            )
            
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id,
            }
    
    async def _execute_http_tool(
        self,
        tool: RegisteredTool,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute HTTP tool."""
        try:
            import httpx
            
            http_config = tool.config.get("http", {})
            url = http_config.get("url")
            method = http_config.get("method", "POST").upper()
            headers = http_config.get("headers", {})
            
            async with httpx.AsyncClient() as client:
                if method == "GET":
                    response = await client.get(url, params=input_data, headers=headers)
                else:
                    response = await client.post(url, json=input_data, headers=headers)
                
                response.raise_for_status()
                
                return {
                    "success": True,
                    "result": response.json(),
                    "tool_id": tool.tool_id,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool.tool_id,
            }
    
    async def _execute_mcp_tool(
        self,
        tool: RegisteredTool,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute MCP tool."""
        return {
            "success": False,
            "error": "MCP tool execution not yet implemented",
            "tool_id": tool.tool_id,
        }
    
    def get_tool_schemas(self) -> list:
        """Get tool schemas for LLM."""
        return self.registry.get_tool_schemas()
