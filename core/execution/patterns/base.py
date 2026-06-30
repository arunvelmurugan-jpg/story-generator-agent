"""
Base Pattern for PHTN.AI Sub-Agent Framework

Abstract base class for all execution patterns.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from ...config_loader import ExecutionConfig
    from ....llm.router import LLMRouter
    from ....tools.executor import ToolExecutor
    from ....memory.manager import MemoryManager
    from ....guardrails.engine import GuardrailsEngine
    from ....context.builder import ContextBuilder
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStep:
    """Represents a step in the execution trace."""
    step_number: int
    step_type: str
    input_data: Any
    output_data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "input": self.input_data,
            "output": self.output_data,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionContext:
    """Context for pattern execution."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    tools_available: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: Optional[str] = None
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    few_shot_examples: List[Dict[str, Any]] = field(default_factory=list)
    prompt_variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the context."""
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.messages)
        return messages


class BasePattern(ABC):
    """
    Abstract base class for execution patterns.
    
    All execution patterns must implement:
    - execute(): Main execution method
    - execute_stream(): Streaming execution method
    
    Patterns can optionally override:
    - prepare_context(): Prepare execution context
    - post_process(): Post-process the output
    """
    
    pattern_name: str = "base"
    
    def __init__(
        self,
        config: "ExecutionConfig",
        llm_client: "LLMRouter",
        tool_executor: "ToolExecutor",
        memory_manager: "MemoryManager",
        guardrails_engine: "GuardrailsEngine",
        context_builder: "ContextBuilder",
    ):
        """
        Initialize BasePattern.
        
        Args:
            config: Execution configuration
            llm_client: LLM router instance
            tool_executor: Tool executor instance
            memory_manager: Memory manager instance
            guardrails_engine: Guardrails engine instance
            context_builder: Context builder instance
        """
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory_manager = memory_manager
        self.guardrails_engine = guardrails_engine
        self.context_builder = context_builder
        
        self._execution_trace: List[ExecutionStep] = []
        self._step_counter = 0
        
        logger.debug(f"Pattern initialized: {self.pattern_name}")
    
    @abstractmethod
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """
        Execute the pattern with given input.
        
        Args:
            input_data: Agent input
            context: Execution context
            
        Returns:
            AgentOutput with results
        """
        pass
    
    @abstractmethod
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute the pattern with streaming output.
        
        Args:
            input_data: Agent input
            context: Execution context
            
        Yields:
            Streaming chunks
        """
        pass
    
    async def prepare_context(
        self,
        input_data: "AgentInput",
    ) -> ExecutionContext:
        """
        Prepare execution context from input.
        
        Args:
            input_data: Agent input
            
        Returns:
            ExecutionContext
        """
        context = ExecutionContext()
        
        memory_context = await self.memory_manager.get_context(
            session_id=input_data.context.request_id,
        )
        if memory_context:
            context.memory_context = memory_context
        
        tools = self.tool_executor.get_tool_schemas()
        context.tools_available = tools
        
        return context
    
    async def post_process(
        self,
        output: "AgentOutput",
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """
        Post-process the output.
        
        Args:
            output: Raw output
            input_data: Original input
            context: Execution context
            
        Returns:
            Processed AgentOutput
        """
        output.execution_trace = [step.to_dict() for step in self._execution_trace]
        return output
    
    def add_trace_step(
        self,
        step_type: str,
        input_data: Any,
        output_data: Any,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionStep:
        """Add a step to the execution trace."""
        self._step_counter += 1
        step = ExecutionStep(
            step_number=self._step_counter,
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._execution_trace.append(step)
        return step
    
    def reset_trace(self):
        """Reset execution trace."""
        self._execution_trace.clear()
        self._step_counter = 0
    
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Call LLM with messages.
        
        Args:
            messages: Chat messages
            tools: Optional tool definitions
            **kwargs: Additional LLM parameters
            
        Returns:
            LLM response
        """
        start_time = datetime.utcnow()
        
        response = await self.llm_client.complete(
            messages=messages,
            tools=tools,
            **kwargs,
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        self.add_trace_step(
            step_type="llm_call",
            input_data={"messages": messages, "tools": bool(tools)},
            output_data=response,
            duration_ms=duration_ms,
        )
        
        return response
    
    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Tool name/ID
            tool_input: Tool input parameters
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        
        result = await self.tool_executor.execute(
            tool_id=tool_name,
            input_data=tool_input,
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        self.add_trace_step(
            step_type="tool_call",
            input_data={"tool": tool_name, "input": tool_input},
            output_data=result,
            duration_ms=duration_ms,
        )
        
        return result
