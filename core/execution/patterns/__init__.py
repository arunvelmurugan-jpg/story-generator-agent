"""
Execution Patterns for PHTN.AI Sub-Agent Framework

Provides different execution patterns for agent behavior:
- SimplePattern: Direct LLM call without tools
- ReactPattern: Reasoning and Acting loop
- ChainOfThoughtPattern: Step-by-step reasoning
- ToolUsePattern: Tool-calling focused execution
- RAGPattern: Retrieval-Augmented Generation
- PlanExecutePattern: Planning then execution
- CustomPattern: User-defined patterns
"""

from .base import BasePattern
from .simple import SimplePattern
from .react import ReactPattern
from .cot import ChainOfThoughtPattern
from .tool_use import ToolUsePattern
from .rag import RAGPattern
from .plan_execute import PlanExecutePattern
from .custom import CustomPattern

__all__ = [
    "BasePattern",
    "SimplePattern",
    "ReactPattern",
    "ChainOfThoughtPattern",
    "ToolUsePattern",
    "RAGPattern",
    "PlanExecutePattern",
    "CustomPattern",
]
