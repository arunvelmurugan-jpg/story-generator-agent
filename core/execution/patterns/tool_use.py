"""
Tool Use Execution Pattern for PHTN.AI Sub-Agent Framework

Implements tool-calling focused execution using native LLM tool calling.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class ToolUsePattern(BasePattern):
    """
    Tool Use execution pattern.
    
    This pattern leverages native LLM tool calling capabilities:
    1. Send user request with tool definitions
    2. LLM decides which tools to call (if any)
    3. Execute tool calls
    4. Return results to LLM for final response
    
    Best for:
    - Tasks requiring external data/APIs
    - Function calling scenarios
    - Structured data extraction
    - Multi-tool orchestration
    """
    
    pattern_name = "tool_use"
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute Tool Use pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        messages = self._build_messages(input_data, context)
        tools = self._format_tools_for_llm(context.tools_available)
        
        tool_calls_made = []
        iteration = 0
        max_iterations = self.config.max_iterations
        
        while iteration < max_iterations:
            iteration += 1
            
            response = await self.call_llm(messages, tools=tools)
            
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                content = response.get("content", "")
                
                output = AgentOutput(
                    content=content,
                    content_type=ContentType.TEXT,
                    success=True,
                    tool_calls=tool_calls_made,
                    token_usage=response.get("usage", {}),
                    metadata={"iterations": iteration},
                )
                
                return await self.post_process(output, input_data, context)
            
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": tool_calls,
            })
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                tool_call_id = tool_call.get("id")
                
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"input": tool_args}
                
                try:
                    result = await self.execute_tool(tool_name, tool_args)
                    tool_result = result.get("result", str(result))
                    tool_calls_made.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": tool_result,
                        "success": True,
                    })
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                    tool_calls_made.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": tool_result,
                        "success": False,
                        "error": str(e),
                    })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(tool_result),
                })
        
        output = AgentOutput(
            content="Maximum iterations reached without final response.",
            content_type=ContentType.TEXT,
            success=False,
            tool_calls=tool_calls_made,
            error="Max iterations reached",
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute Tool Use pattern with streaming."""
        self.reset_trace()
        
        messages = self._build_messages(input_data, context)
        tools = self._format_tools_for_llm(context.tools_available)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        tool_calls_made = []
        iteration = 0
        max_iterations = self.config.max_iterations
        
        while iteration < max_iterations:
            iteration += 1
            
            response = await self.call_llm(messages, tools=tools)
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                content = response.get("content", "")
                
                for i in range(0, len(content), 50):
                    yield {
                        "type": "content",
                        "content": content[i:i+50],
                    }
                
                yield {
                    "type": "end",
                    "tool_calls": tool_calls_made,
                    "iterations": iteration,
                }
                return
            
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": tool_calls,
            })
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                tool_call_id = tool_call.get("id")
                
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"input": tool_args}
                
                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_args,
                }
                
                try:
                    result = await self.execute_tool(tool_name, tool_args)
                    tool_result = result.get("result", str(result))
                    tool_calls_made.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": tool_result,
                        "success": True,
                    })
                    
                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": tool_result,
                        "success": True,
                    }
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                    tool_calls_made.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": tool_result,
                        "success": False,
                    })
                    
                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": tool_result,
                        "success": False,
                        "error": str(e),
                    }
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(tool_result),
                })
        
        yield {
            "type": "end",
            "error": "Max iterations reached",
            "tool_calls": tool_calls_made,
        }
    
    def _build_messages(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Build messages for tool use."""
        messages = []
        
        system_prompt = context.system_prompt or "You are a helpful AI assistant with access to tools."
        
        if context.memory_context:
            system_prompt += f"\n\nRelevant context:\n{context.memory_context}"
        
        messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        
        messages.append({"role": "user", "content": str(user_content)})
        
        return messages
    
    def _format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for LLM tool calling API."""
        formatted = []
        
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", tool.get("tool_id")),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", tool.get("input_schema", {
                        "type": "object",
                        "properties": {},
                    })),
                },
            })
        
        return formatted
