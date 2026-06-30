"""
Simple Execution Pattern for PHTN.AI Sub-Agent Framework

Direct LLM call without tool usage - the simplest execution pattern.
"""

import logging
from datetime import datetime
from typing import Any, Dict, AsyncIterator, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class SimplePattern(BasePattern):
    """
    Simple execution pattern - direct LLM call.
    
    This pattern:
    1. Prepares context with system prompt and memory
    2. Calls LLM with user input
    3. Returns the response
    
    Best for:
    - Simple Q&A tasks
    - Text generation without tools
    - Classification tasks
    - Summarization
    """
    
    pattern_name = "simple"
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute simple pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        messages = self._build_messages(input_data, context)
        
        response = await self.call_llm(messages)
        
        content = response.get("content", "")
        token_usage = response.get("usage", {})
        
        output = AgentOutput(
            content=content,
            content_type=ContentType.TEXT,
            success=True,
            token_usage=token_usage,
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute simple pattern with streaming."""
        self.reset_trace()
        
        messages = self._build_messages(input_data, context)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        full_content = ""
        async for chunk in self.llm_client.stream(messages):
            content_delta = chunk.get("content", "")
            full_content += content_delta
            
            yield {
                "type": "content",
                "content": content_delta,
                "accumulated": full_content,
            }
        
        yield {
            "type": "end",
            "content": full_content,
            "usage": chunk.get("usage", {}),
        }
    
    def _build_messages(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> list:
        """Build messages for LLM call."""
        messages = []
        
        if context.system_prompt:
            system_content = context.system_prompt
            if context.memory_context:
                system_content += f"\n\nContext from memory:\n{context.memory_context}"
            messages.append({"role": "system", "content": system_content})
        elif context.memory_context:
            messages.append({
                "role": "system",
                "content": f"Context from memory:\n{context.memory_context}"
            })
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        
        messages.append({"role": "user", "content": str(user_content)})
        
        return messages
