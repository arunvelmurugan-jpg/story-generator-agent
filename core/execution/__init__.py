"""
Execution Engine Module for PHTN.AI Sub-Agent Framework

Provides execution patterns for agent behavior:
- SIMPLE: Direct LLM call
- REACT: Reasoning and Acting loop
- COT: Chain-of-Thought reasoning
- TOOL_USE: Tool-calling focused execution
- RAG: Retrieval-Augmented Generation
- PLAN_EXECUTE: Planning then execution
- CUSTOM: User-defined patterns
"""

from .engine import ExecutionEngine
from .state_machine import StateMachine, State, Transition
from .patterns.base import BasePattern
from .patterns.simple import SimplePattern
from .patterns.react import ReactPattern
from .patterns.cot import ChainOfThoughtPattern
from .patterns.tool_use import ToolUsePattern
from .patterns.rag import RAGPattern
from .patterns.plan_execute import PlanExecutePattern

__all__ = [
    "ExecutionEngine",
    "StateMachine",
    "State",
    "Transition",
    "BasePattern",
    "SimplePattern",
    "ReactPattern",
    "ChainOfThoughtPattern",
    "ToolUsePattern",
    "RAGPattern",
    "PlanExecutePattern",
]
