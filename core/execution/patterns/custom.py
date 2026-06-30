"""
Custom Execution Pattern for PHTN.AI Sub-Agent Framework

Allows users to define custom execution patterns.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, Callable, Optional, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class CustomPattern(BasePattern):
    """
    Custom execution pattern for user-defined behavior.
    
    This pattern allows users to:
    1. Define custom execution logic via callbacks
    2. Combine elements from other patterns
    3. Implement domain-specific workflows
    
    Best for:
    - Domain-specific agent behavior
    - Experimental patterns
    - Hybrid approaches
    - Integration with external systems
    """
    
    pattern_name = "custom"
    
    _custom_executor: Optional[Callable] = None
    _custom_stream_executor: Optional[Callable] = None
    
    @classmethod
    def register_executor(cls, executor: Callable):
        """Register a custom executor function."""
        cls._custom_executor = executor
    
    @classmethod
    def register_stream_executor(cls, executor: Callable):
        """Register a custom streaming executor function."""
        cls._custom_stream_executor = executor
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute custom pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        custom_config = self.config.custom_config or {}
        
        if self._custom_executor:
            try:
                result = await self._custom_executor(
                    input_data=input_data,
                    context=context,
                    config=custom_config,
                    llm_client=self.llm_client,
                    tool_executor=self.tool_executor,
                    memory_manager=self.memory_manager,
                )
                
                if isinstance(result, AgentOutput):
                    return await self.post_process(result, input_data, context)
                
                return AgentOutput(
                    content=result,
                    content_type=ContentType.JSON_OBJECT if isinstance(result, dict) else ContentType.TEXT,
                    success=True,
                )
            except Exception as e:
                logger.error(f"Custom executor failed: {e}")
                return AgentOutput(
                    content=None,
                    success=False,
                    error=f"Custom executor error: {str(e)}",
                )
        
        return await self._default_execute(input_data, context, custom_config)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute custom pattern with streaming."""
        self.reset_trace()
        
        custom_config = self.config.custom_config or {}
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        if self._custom_stream_executor:
            try:
                async for chunk in self._custom_stream_executor(
                    input_data=input_data,
                    context=context,
                    config=custom_config,
                    llm_client=self.llm_client,
                    tool_executor=self.tool_executor,
                    memory_manager=self.memory_manager,
                ):
                    yield chunk
            except Exception as e:
                yield {"type": "error", "error": str(e)}
        else:
            async for chunk in self._default_execute_stream(input_data, context, custom_config):
                yield chunk
        
        yield {"type": "end"}
    
    async def _default_execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
        config: Dict[str, Any],
    ) -> "AgentOutput":
        """Default execution when no custom executor is registered."""
        from ...agent import AgentOutput, ContentType
        
        use_tools = config.get("use_tools", True)
        use_memory = config.get("use_memory", True)
        
        messages = []
        
        system_prompt = config.get("system_prompt", context.system_prompt)
        if system_prompt:
            if use_memory and context.memory_context:
                system_prompt += f"\n\nContext:\n{context.memory_context}"
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        messages.append({"role": "user", "content": str(user_content)})
        
        tools = context.tools_available if use_tools else None
        
        response = await self.call_llm(messages, tools=tools)
        
        return AgentOutput(
            content=response.get("content", ""),
            content_type=ContentType.TEXT,
            success=True,
            token_usage=response.get("usage", {}),
        )
    
    async def _default_execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
        config: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Default streaming execution."""
        use_memory = config.get("use_memory", True)
        
        messages = []
        
        system_prompt = config.get("system_prompt", context.system_prompt)
        if system_prompt:
            if use_memory and context.memory_context:
                system_prompt += f"\n\nContext:\n{context.memory_context}"
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        messages.append({"role": "user", "content": str(user_content)})
        
        async for chunk in self.llm_client.stream(messages):
            yield {
                "type": "content",
                "content": chunk.get("content", ""),
            }
