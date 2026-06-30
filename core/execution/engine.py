"""
Execution Engine for PHTN.AI Sub-Agent Framework

The ExecutionEngine orchestrates agent execution by:
- Selecting the appropriate execution pattern
- Managing the execution lifecycle
- Coordinating LLM, tools, memory, and guardrails
- Handling streaming and non-streaming execution
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, AsyncIterator, TYPE_CHECKING

from ..config_loader import ExecutionConfig, ExecutionPattern

if TYPE_CHECKING:
    from ..agent import AgentInput, AgentOutput
    from ...llm.router import LLMRouter
    from ...tools.executor import ToolExecutor
    from ...memory.manager import MemoryManager
    from ...guardrails.engine import GuardrailsEngine
    from ...context.builder import ContextBuilder

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Orchestrates agent execution across different patterns.
    
    The ExecutionEngine is responsible for:
    - Pattern selection and instantiation
    - Execution lifecycle management
    - Resource coordination (LLM, tools, memory)
    - Error handling and recovery
    - Execution tracing and metrics
    """
    
    def __init__(
        self,
        pattern: ExecutionPattern,
        config: ExecutionConfig,
        llm_client: "LLMRouter",
        tool_executor: "ToolExecutor",
        memory_manager: "MemoryManager",
        guardrails_engine: "GuardrailsEngine",
        context_builder: "ContextBuilder",
    ):
        """
        Initialize ExecutionEngine.
        
        Args:
            pattern: Execution pattern to use
            config: Execution configuration
            llm_client: LLM router instance
            tool_executor: Tool executor instance
            memory_manager: Memory manager instance
            guardrails_engine: Guardrails engine instance
            context_builder: Context builder instance
        """
        self.pattern = pattern
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.memory_manager = memory_manager
        self.guardrails_engine = guardrails_engine
        self.context_builder = context_builder
        
        self._pattern_instance = self._create_pattern()
        
        logger.info(f"ExecutionEngine initialized with pattern: {pattern.value}")
    
    def _create_pattern(self):
        """Create pattern instance based on configuration."""
        from .patterns.simple import SimplePattern
        from .patterns.react import ReactPattern
        from .patterns.cot import ChainOfThoughtPattern
        from .patterns.tool_use import ToolUsePattern
        from .patterns.rag import RAGPattern
        from .patterns.plan_execute import PlanExecutePattern
        from .patterns.custom import CustomPattern
        
        pattern_map = {
            ExecutionPattern.SIMPLE: SimplePattern,
            ExecutionPattern.REACT: ReactPattern,
            ExecutionPattern.COT: ChainOfThoughtPattern,
            ExecutionPattern.TOOL_USE: ToolUsePattern,
            ExecutionPattern.RAG: RAGPattern,
            ExecutionPattern.PLAN_EXECUTE: PlanExecutePattern,
            ExecutionPattern.CUSTOM: CustomPattern,
        }
        
        pattern_class = pattern_map.get(self.pattern, SimplePattern)
        
        return pattern_class(
            config=self.config,
            llm_client=self.llm_client,
            tool_executor=self.tool_executor,
            memory_manager=self.memory_manager,
            guardrails_engine=self.guardrails_engine,
            context_builder=self.context_builder,
        )
    
    async def execute(self, input_data: "AgentInput") -> "AgentOutput":
        """
        Execute agent with given input.
        
        Args:
            input_data: Agent input
            
        Returns:
            AgentOutput with results
        """
        from ..agent import AgentOutput
        
        start_time = datetime.utcnow()
        
        try:
            context = await self.context_builder.build(input_data)
            
            await self.memory_manager.store_input(
                session_id=input_data.context.request_id,
                content=input_data.content,
            )
            
            output = await self._pattern_instance.execute(input_data, context)
            
            await self.memory_manager.store_output(
                session_id=input_data.context.request_id,
                content=output.content,
            )
            
            end_time = datetime.utcnow()
            output.latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return output
            
        except asyncio.TimeoutError:
            logger.error(f"Execution timeout after {self.config.timeout_seconds}s")
            return AgentOutput(
                content=None,
                success=False,
                error=f"Execution timeout after {self.config.timeout_seconds} seconds",
            )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return AgentOutput(
                content=None,
                success=False,
                error=str(e),
            )
    
    async def execute_stream(self, input_data: "AgentInput") -> AsyncIterator[Dict[str, Any]]:
        """
        Execute agent with streaming output.
        
        Args:
            input_data: Agent input
            
        Yields:
            Streaming chunks
        """
        try:
            context = await self.context_builder.build(input_data)
            
            await self.memory_manager.store_input(
                session_id=input_data.context.request_id,
                content=input_data.content,
            )
            
            async for chunk in self._pattern_instance.execute_stream(input_data, context):
                yield chunk
                
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "error": f"Execution timeout after {self.config.timeout_seconds} seconds"
            }
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """Get information about the current pattern."""
        return {
            "pattern": self.pattern.value,
            "max_iterations": self.config.max_iterations,
            "timeout_seconds": self.config.timeout_seconds,
            "pattern_config": self._get_pattern_specific_config(),
        }
    
    def _get_pattern_specific_config(self) -> Optional[Dict[str, Any]]:
        """Get pattern-specific configuration."""
        config_map = {
            ExecutionPattern.REACT: self.config.react_config,
            ExecutionPattern.COT: self.config.cot_config,
            ExecutionPattern.RAG: self.config.rag_config,
            ExecutionPattern.PLAN_EXECUTE: self.config.plan_execute_config,
            ExecutionPattern.CUSTOM: self.config.custom_config,
        }
        return config_map.get(self.pattern)
