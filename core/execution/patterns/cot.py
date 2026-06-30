"""
Chain-of-Thought Execution Pattern for PHTN.AI Sub-Agent Framework

Implements explicit step-by-step reasoning before providing an answer.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class ChainOfThoughtPattern(BasePattern):
    """
    Chain-of-Thought (CoT) execution pattern.
    
    This pattern encourages the LLM to:
    1. Break down the problem into steps
    2. Reason through each step explicitly
    3. Arrive at a final answer
    
    Best for:
    - Complex reasoning tasks
    - Math problems
    - Logic puzzles
    - Multi-step analysis
    """
    
    pattern_name = "cot"
    
    COT_SYSTEM_PROMPT = """You are an AI assistant that uses Chain-of-Thought reasoning.

When answering questions or solving problems:
1. Break down the problem into clear steps
2. Think through each step carefully
3. Show your reasoning process
4. Arrive at a well-reasoned conclusion

Format your response as:
## Step-by-Step Reasoning

**Step 1:** [First step of reasoning]
**Step 2:** [Second step of reasoning]
...

## Conclusion

[Your final answer based on the reasoning above]

Be thorough in your reasoning and make sure each step logically follows from the previous one.
"""
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute Chain-of-Thought pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        cot_config = self.config.cot_config or {}
        explicit_reasoning = cot_config.get("explicit_reasoning", True)
        step_by_step = cot_config.get("step_by_step", True)
        
        messages = self._build_messages(input_data, context, explicit_reasoning, step_by_step)
        
        response = await self.call_llm(messages)
        content = response.get("content", "")
        
        reasoning_steps, conclusion = self._parse_cot_response(content)
        
        output = AgentOutput(
            content=conclusion or content,
            content_type=ContentType.TEXT,
            success=True,
            token_usage=response.get("usage", {}),
            metadata={
                "reasoning_steps": reasoning_steps,
                "full_response": content,
            },
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute Chain-of-Thought pattern with streaming."""
        self.reset_trace()
        
        cot_config = self.config.cot_config or {}
        explicit_reasoning = cot_config.get("explicit_reasoning", True)
        step_by_step = cot_config.get("step_by_step", True)
        
        messages = self._build_messages(input_data, context, explicit_reasoning, step_by_step)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        full_content = ""
        current_section = "reasoning"
        
        async for chunk in self.llm_client.stream(messages):
            content_delta = chunk.get("content", "")
            full_content += content_delta
            
            if "## Conclusion" in full_content and current_section == "reasoning":
                current_section = "conclusion"
                yield {"type": "section_change", "section": "conclusion"}
            
            yield {
                "type": "content",
                "section": current_section,
                "content": content_delta,
            }
        
        reasoning_steps, conclusion = self._parse_cot_response(full_content)
        
        yield {
            "type": "end",
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
        }
    
    def _build_messages(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
        explicit_reasoning: bool,
        step_by_step: bool,
    ) -> List[Dict[str, Any]]:
        """Build messages for CoT."""
        system_prompt = self.COT_SYSTEM_PROMPT
        
        if context.system_prompt:
            system_prompt = context.system_prompt + "\n\n" + system_prompt
        
        if context.memory_context:
            system_prompt += f"\n\nRelevant context:\n{context.memory_context}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        
        if step_by_step:
            user_content = f"{user_content}\n\nPlease think through this step by step."
        
        messages.append({"role": "user", "content": str(user_content)})
        
        return messages
    
    def _parse_cot_response(self, response: str) -> tuple:
        """Parse CoT response to extract reasoning steps and conclusion."""
        import re
        
        reasoning_steps = []
        conclusion = None
        
        step_pattern = r"\*\*Step (\d+):\*\*\s*(.+?)(?=\*\*Step|\#\#|$)"
        steps = re.findall(step_pattern, response, re.DOTALL)
        
        for step_num, step_content in steps:
            reasoning_steps.append({
                "step": int(step_num),
                "content": step_content.strip(),
            })
        
        conclusion_match = re.search(r"##\s*Conclusion\s*\n+(.+?)$", response, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        
        return reasoning_steps, conclusion
