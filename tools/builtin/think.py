"""
Think Tool - Agent Reasoning

Allows the agent to reason and reflect before taking action.
Equivalent to n8n's Think tool.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ThinkResult:
    """Result of a thinking step."""
    thought: str
    reasoning_type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThinkTool:
    """
    Think tool for agent reasoning and reflection.
    
    This tool allows the agent to:
    - Reason about the current situation
    - Plan next steps
    - Reflect on previous actions
    - Consider alternatives
    - Validate assumptions
    
    The output is not returned to the user but helps
    the agent make better decisions.
    """
    
    name = "think"
    description = """Use this tool to think through a problem step by step.
The thought will not be shown to the user - it's for your internal reasoning.
Use this when you need to:
- Plan a complex task
- Consider multiple approaches
- Validate your assumptions
- Reflect on previous steps
- Reason about edge cases"""
    
    REASONING_TYPES = [
        "planning",
        "reflection",
        "analysis",
        "validation",
        "consideration",
        "hypothesis",
        "conclusion",
        "question",
    ]
    
    def __init__(self):
        self.thought_history: List[ThinkResult] = []
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning or thought process"
                    },
                    "reasoning_type": {
                        "type": "string",
                        "enum": self.REASONING_TYPES,
                        "description": "Type of reasoning being performed"
                    }
                },
                "required": ["thought"]
            }
        }
    
    async def execute(
        self,
        thought: str,
        reasoning_type: str = "analysis"
    ) -> ThinkResult:
        """
        Record a thinking step.
        
        Args:
            thought: The thought or reasoning
            reasoning_type: Type of reasoning
        
        Returns:
            ThinkResult with the recorded thought
        """
        result = ThinkResult(
            thought=thought,
            reasoning_type=reasoning_type,
            metadata={
                "thought_index": len(self.thought_history),
                "previous_thoughts": len(self.thought_history)
            }
        )
        
        self.thought_history.append(result)
        
        logger.debug(f"Think [{reasoning_type}]: {thought[:100]}...")
        
        return result
    
    def get_thought_summary(self) -> str:
        """Get a summary of all thoughts."""
        if not self.thought_history:
            return "No thoughts recorded yet."
        
        summary_parts = []
        for i, thought in enumerate(self.thought_history):
            summary_parts.append(
                f"{i+1}. [{thought.reasoning_type}] {thought.thought[:100]}..."
            )
        
        return "\n".join(summary_parts)
    
    def get_thoughts_by_type(self, reasoning_type: str) -> List[ThinkResult]:
        """Get all thoughts of a specific type."""
        return [t for t in self.thought_history if t.reasoning_type == reasoning_type]
    
    def clear_history(self):
        """Clear thought history."""
        self.thought_history = []
    
    def __call__(self, thought: str, reasoning_type: str = "analysis") -> str:
        """Synchronous think - returns acknowledgment."""
        result = ThinkResult(
            thought=thought,
            reasoning_type=reasoning_type
        )
        self.thought_history.append(result)
        return f"Thought recorded: {thought[:50]}..."


class ReflectionTool(ThinkTool):
    """
    Specialized thinking tool for reflection.
    
    Helps the agent reflect on:
    - What went well
    - What could be improved
    - Lessons learned
    - Next steps
    """
    
    name = "reflect"
    description = """Use this tool to reflect on your actions and their outcomes.
Reflection helps improve future performance by:
- Analyzing what worked and what didn't
- Identifying patterns in successes and failures
- Planning improvements for next time"""
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action_taken": {
                        "type": "string",
                        "description": "The action you took"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "What happened as a result"
                    },
                    "what_worked": {
                        "type": "string",
                        "description": "What went well"
                    },
                    "what_to_improve": {
                        "type": "string",
                        "description": "What could be done better"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "What to do next based on this reflection"
                    }
                },
                "required": ["action_taken", "outcome"]
            }
        }
    
    async def execute(
        self,
        action_taken: str,
        outcome: str,
        what_worked: str = "",
        what_to_improve: str = "",
        next_steps: str = ""
    ) -> ThinkResult:
        """Record a reflection."""
        thought = f"""
Action: {action_taken}
Outcome: {outcome}
What worked: {what_worked}
To improve: {what_to_improve}
Next steps: {next_steps}
""".strip()
        
        return await super().execute(thought, "reflection")


__all__ = ["ThinkTool", "ThinkResult", "ReflectionTool"]
