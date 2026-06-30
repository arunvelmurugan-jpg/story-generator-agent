"""
Main Agent Class for PHTN.AI Sub-Agent Framework

The Agent class is the core component that:
- Loads configuration from PHTN-AGENT.json
- Selects and executes the appropriate execution pattern
- Manages tools, memory, and context
- Enforces guardrails and security policies
- Provides observability through X-PHTN-Agent-ID header

Implements OTEL-compatible logging with full X-PHTN-Agent-ID support.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator

from .config_loader import (
    ConfigLoader,
    AgentConfiguration,
    ExecutionPattern,
    ContentType,
    CapabilityDefinition,
    SkillDefinition,
    ToolDefinition,
)
from ..observability.otel_logging import (
    get_logger,
    set_trace_context,
    set_trace_context_from_config,
    get_trace_context,
    log_agent_execution,
)

logger = get_logger(__name__)


class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentContext:
    """Context for agent execution."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = "development"
    capability_id: Optional[str] = None
    skill_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_phtn_agent_id(self, config: AgentConfiguration) -> str:
        """Generate X-PHTN-Agent-ID header value."""
        parts = [
            config.tenant or "",
            config.tenant or "",
            config.domain.industry if config.domain else "",
            "",
            "",
            self.user_id or "",
            "SUB-AGENT",
            "",
            config.agent_id,
            "",
            config.instance_id or "",
            config.name,
            self.environment,
            self.correlation_id or self.request_id,
            self.trace_id or "",
            self.span_id or "",
            self.capability_id or "",
            self.skill_id or "",
        ]
        return "|".join(parts)


@dataclass
class AgentInput:
    """Input to agent execution."""
    content: Any
    content_type: ContentType = ContentType.JSON_OBJECT
    context: Optional[AgentContext] = None
    capability_id: Optional[str] = None
    skill_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.context is None:
            self.context = AgentContext()
        if self.capability_id:
            self.context.capability_id = self.capability_id
        if self.skill_id:
            self.context.skill_id = self.skill_id


@dataclass
class AgentOutput:
    """Output from agent execution."""
    content: Any
    content_type: ContentType = ContentType.JSON_OBJECT
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    
    def add_trace(self, step: str, data: Dict[str, Any]):
        """Add execution trace step."""
        self.execution_trace.append({
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })


@dataclass
class AgentConfig:
    """Runtime agent configuration."""
    config_path: Optional[Path] = None
    override_model: Optional[str] = None
    override_temperature: Optional[float] = None
    override_max_tokens: Optional[int] = None
    enable_streaming: bool = False
    enable_tracing: bool = True
    enable_metrics: bool = True
    dry_run: bool = False


class Agent:
    """
    PHTN.AI Enterprise Sub-Agent.
    
    The Agent class provides a unified interface for executing AI agent tasks
    with support for multiple execution patterns, tools, memory, guardrails,
    and full observability.
    
    Features:
    - Multiple execution patterns (SIMPLE, REACT, COT, TOOL_USE, RAG, PLAN_EXECUTE)
    - Multi-provider LLM support with routing and fallbacks
    - Tool execution with sandboxing
    - Memory management (short-term, long-term, semantic, episodic)
    - Comprehensive guardrails (input/output/tool)
    - RBAC/ABAC security
    - Full observability with X-PHTN-Agent-ID header
    - A2A protocol support
    - MCP compatibility
    """
    
    def __init__(
        self,
        config: Optional[AgentConfiguration] = None,
        config_path: Optional[Path] = None,
        runtime_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize Agent.
        
        Args:
            config: Pre-loaded AgentConfiguration
            config_path: Path to PHTN-AGENT.json (if config not provided)
            runtime_config: Runtime configuration overrides
        """
        self.runtime_config = runtime_config or AgentConfig()
        
        if config:
            self._config = config
        else:
            loader = ConfigLoader(
                phtnai_dir=config_path.parent if config_path else None
            )
            self._config = loader.load_agent_config(config_path)
        
        self._state = AgentState.IDLE
        self._execution_engine = None
        self._llm_client = None
        self._tool_executor = None
        self._memory_manager = None
        self._guardrails_engine = None
        self._context_builder = None
        
        self._initialized = False
        
        # Set comprehensive trace context from config for phtnai-ops-metrics
        set_trace_context_from_config(self._config)
        
        # Also set additional context that may not be in config
        set_trace_context(
            agent_id=self._config.agent_id,
            agent_name=self._config.name,
            tenant_id=self._config.tenant or "",
            tenant_group_id=self._config.tenant or "",
            domain_id=self._config.domain.industry if self._config.domain else "unknown",
            agent_type="SUB_AGENT",
            sub_agent_id=self._config.agent_id,
            app_name=self._config.name,
            environment=self._config.deployment_metadata.environment if self._config.deployment_metadata else "development",
        )
        
        logger.info(f"🤖 Agent created: {self._config.name} (id={self._config.agent_id})")
    
    @property
    def config(self) -> AgentConfiguration:
        """Get agent configuration."""
        return self._config
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._config.agent_id
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self._config.name
    
    @property
    def execution_pattern(self) -> ExecutionPattern:
        """Get execution pattern."""
        if self._config.execution_config:
            return self._config.execution_config.pattern
        return ExecutionPattern.SIMPLE
    
    @property
    def capabilities(self) -> List[CapabilityDefinition]:
        """Get agent capabilities."""
        return self._config.capabilities
    
    @property
    def tools(self) -> List[ToolDefinition]:
        """Get configured tools."""
        return self._config.tools
    
    async def initialize(self) -> None:
        """
        Initialize agent components.
        
        This method initializes:
        - LLM client with configured provider
        - Tool executor with registered tools
        - Memory manager
        - Guardrails engine
        - Context builder
        - Execution engine for the configured pattern
        """
        if self._initialized:
            logger.warning("⚠️ Agent already initialized")
            return
        
        self._state = AgentState.INITIALIZING
        logger.info(f"🔧 Initializing agent: {self.name}")
        
        try:
            await self._initialize_llm_client()
            await self._initialize_memory_manager()
            await self._initialize_guardrails_engine()
            await self._initialize_tool_executor()
            await self._initialize_context_builder()
            await self._initialize_execution_engine()
            
            self._initialized = True
            self._state = AgentState.READY
            logger.info(f"✅ Agent initialized successfully: {self.name}")
            
        except Exception as e:
            self._state = AgentState.ERROR
            logger.error(f"❌ Failed to initialize agent: {e}", exc_info=True)
            raise
    
    async def _initialize_llm_client(self) -> None:
        """Initialize LLM client."""
        from ..llm.router import LLMRouter
        
        llm_config = self._config.llm_config
        if llm_config is None:
            logger.warning("No LLM config found, using defaults")
            self._llm_client = LLMRouter(
                primary_model="gpt-4",
                provider="openai",
            )
            return
        
        azure_config = None
        if hasattr(llm_config, 'azure_config') and llm_config.azure_config:
            azure_config = llm_config.azure_config.model_dump() if hasattr(llm_config.azure_config, 'model_dump') else llm_config.azure_config
        
        vision_config = None
        if hasattr(llm_config, 'vision') and llm_config.vision:
            vision_config = llm_config.vision.model_dump() if hasattr(llm_config.vision, 'model_dump') else llm_config.vision
        
        audio_config = None
        if hasattr(llm_config, 'audio') and llm_config.audio:
            audio_config = llm_config.audio.model_dump() if hasattr(llm_config.audio, 'model_dump') else llm_config.audio
        
        structured_output_config = None
        if hasattr(llm_config, 'structured_output') and llm_config.structured_output:
            structured_output_config = llm_config.structured_output.model_dump() if hasattr(llm_config.structured_output, 'model_dump') else llm_config.structured_output
        
        embeddings_config = None
        if hasattr(llm_config, 'embeddings') and llm_config.embeddings:
            embeddings_config = llm_config.embeddings.model_dump() if hasattr(llm_config.embeddings, 'model_dump') else llm_config.embeddings
        
        moderation_config = None
        if hasattr(llm_config, 'moderation') and llm_config.moderation:
            moderation_config = llm_config.moderation.model_dump() if hasattr(llm_config.moderation, 'model_dump') else llm_config.moderation
        
        streaming_config = None
        if hasattr(llm_config, 'streaming') and llm_config.streaming:
            streaming_config = llm_config.streaming.model_dump() if hasattr(llm_config.streaming, 'model_dump') else llm_config.streaming
        
        self._llm_client = LLMRouter(
            primary_model=llm_config.primary_model,
            provider=llm_config.provider,
            fallback_models=llm_config.fallback_models,
            parameters=llm_config.parameters.model_dump() if llm_config.parameters else {},
            routing_config=llm_config.routing.model_dump() if llm_config.routing else None,
            azure_config=azure_config,
            vision_config=vision_config,
            audio_config=audio_config,
            structured_output_config=structured_output_config,
            embeddings_config=embeddings_config,
            moderation_config=moderation_config,
            streaming_config=streaming_config,
        )
        
        if self.runtime_config.override_model:
            self._llm_client.set_model(self.runtime_config.override_model)
        if self.runtime_config.override_temperature is not None:
            self._llm_client.set_temperature(self.runtime_config.override_temperature)
        if self.runtime_config.override_max_tokens is not None:
            self._llm_client.set_max_tokens(self.runtime_config.override_max_tokens)
        
        logger.debug(f"LLM client initialized: {llm_config.primary_model}")
    
    async def _initialize_tool_executor(self) -> None:
        """Initialize tool executor with guardrails integration."""
        from ..tools.executor import ToolExecutor
        from ..tools.registry import ToolRegistry
        
        registry = ToolRegistry()
        
        for tool in self._config.tools:
            registry.register(tool)
        
        self._tool_executor = ToolExecutor(
            registry=registry,
            sandboxing_config=self._config.security.tool_sandboxing if self._config.security else None,
            guardrails_engine=self._guardrails_engine,
        )
        
        logger.debug(f"Tool executor initialized with {len(self._config.tools)} tools and guardrails")
    
    async def _initialize_memory_manager(self) -> None:
        """Initialize memory manager."""
        from ..memory.manager import MemoryManager
        
        memory_config = self._config.memory_config
        if memory_config:
            self._memory_manager = MemoryManager(
                short_term_config=memory_config.short_term,
                long_term_config=memory_config.long_term,
                semantic_config=memory_config.semantic,
                episodic_config=memory_config.episodic,
                token_budget=memory_config.token_budget,
            )
        else:
            self._memory_manager = MemoryManager()
        
        logger.debug("Memory manager initialized")
    
    async def _initialize_guardrails_engine(self) -> None:
        """Initialize guardrails engine."""
        from ..guardrails.engine import GuardrailsEngine
        
        self._guardrails_engine = GuardrailsEngine(
            config=self._config.guardrails if self._config.guardrails else None,
        )
        
        logger.debug("Guardrails engine initialized")
    
    async def _initialize_context_builder(self) -> None:
        """Initialize context builder with prompt_config for system_prompt injection."""
        from ..context.builder import ContextBuilder
        
        context_config = self._config.context_strategy
        self._context_builder = ContextBuilder(
            config=context_config,
            memory_manager=self._memory_manager,
            prompt_config=self._config.prompt_config,
            agent_config=self._config,
        )
        
        logger.debug("Context builder initialized with prompt_config")
    
    async def _initialize_execution_engine(self) -> None:
        """Initialize execution engine based on pattern."""
        from .execution.engine import ExecutionEngine
        
        exec_config = self._config.execution_config
        pattern = exec_config.pattern if exec_config else ExecutionPattern.SIMPLE
        
        self._execution_engine = ExecutionEngine(
            pattern=pattern,
            config=exec_config,
            llm_client=self._llm_client,
            tool_executor=self._tool_executor,
            memory_manager=self._memory_manager,
            guardrails_engine=self._guardrails_engine,
            context_builder=self._context_builder,
        )
        
        logger.debug(f"Execution engine initialized: {pattern.value}")
    
    async def execute(
        self,
        input_data: Union[AgentInput, Dict[str, Any], str],
        capability_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> AgentOutput:
        """
        Execute agent with given input.
        
        Supports all 34 content types from PHTNAI_ORCHESTRATION_SCHEMA_v8:
        - Text, JSON, Markdown, HTML, Code
        - Multimodal (images, audio, video)
        - Binary data with base64/gzip encoding
        - Streaming inputs (delta updates)
        - Artifacts and file references
        
        Args:
            input_data: Input data (AgentInput, dict, or string)
            capability_id: Optional capability to invoke
            skill_id: Optional skill to invoke
            context: Optional execution context
            
        Returns:
            AgentOutput with results and content type metadata
        """
        from .content_processor import get_content_processor
        from .content_types import is_multimodal_input
        
        if not self._initialized:
            await self.initialize()
        
        content_processor = get_content_processor()
        
        if isinstance(input_data, str):
            agent_input = AgentInput(
                content=input_data,
                content_type=ContentType.TEXT,
                context=context,
                capability_id=capability_id,
                skill_id=skill_id,
            )
        elif isinstance(input_data, dict):
            # Detect content type from dict
            if is_multimodal_input(input_data):
                # Multimodal input - process parts
                content_type = ContentType.JSON_OBJECT
                logger.info(f"📥 Processing multimodal input with {len(input_data.get('parts', []))} parts")
            elif input_data.get("contentType"):
                # Content envelope - extract type
                try:
                    content_type = ContentType(input_data.get("contentType"))
                except ValueError:
                    content_type = ContentType.JSON_OBJECT
            else:
                # Process input to detect type
                result = content_processor.process_input(input_data)
                content_type = result.content_type
            
            agent_input = AgentInput(
                content=input_data,
                content_type=content_type,
                context=context,
                capability_id=capability_id,
                skill_id=skill_id,
            )
        else:
            agent_input = input_data
            if context:
                agent_input.context = context
            if capability_id:
                agent_input.capability_id = capability_id
            if skill_id:
                agent_input.skill_id = skill_id
        
        start_time = datetime.utcnow()
        self._state = AgentState.EXECUTING
        
        try:
            input_result = await self._guardrails_engine.check_input(agent_input)
            if not input_result.passed:
                return AgentOutput(
                    content=None,
                    success=False,
                    error=f"Input guardrail failed: {input_result.reason}",
                    metadata={"guardrail_result": input_result.to_dict()}
                )
            
            output = await self._execution_engine.execute(agent_input)
            
            output_result = await self._guardrails_engine.check_output(output)
            if not output_result.passed:
                enforcement_mode = self._config.guardrails.enforcement_mode if self._config.guardrails else "warn"
                if enforcement_mode == "enforce":
                    return AgentOutput(
                        content=None,
                        success=False,
                        error=f"Output guardrail failed: {output_result.reason}",
                        metadata={"guardrail_result": output_result.to_dict()}
                    )
                else:
                    output.metadata["guardrail_warning"] = output_result.to_dict()
            
            end_time = datetime.utcnow()
            output.latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Log agent execution for phtnai-ops-metrics AIOps tracking
            exec_pattern = self._config.execution_config.pattern.value if self._config.execution_config else "simple"
            log_agent_execution(
                logger,
                f"Agent execution completed: {self._config.name}",
                execution_pattern=exec_pattern,
                latency_ms=int(output.latency_ms),
                status=200 if output.success else 500,
                iterations=output.metadata.get("iterations", 0),
                tool_calls=len(output.tool_calls),
                capability_id=agent_input.capability_id or "unknown",
                skill_id=agent_input.skill_id or "unknown",
            )
            
            self._state = AgentState.COMPLETED
            return output
            
        except Exception as e:
            self._state = AgentState.ERROR
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Log agent execution failure for phtnai-ops-metrics AIOps tracking
            exec_pattern = self._config.execution_config.pattern.value if self._config.execution_config else "simple"
            log_agent_execution(
                logger,
                f"Agent execution failed: {self._config.name}",
                execution_pattern=exec_pattern,
                latency_ms=latency_ms,
                status=500,
                iterations=0,
                tool_calls=0,
                error=str(e),
            )
            
            logger.error(f"❌ Agent execution failed: {e}", exc_info=True)
            return AgentOutput(
                content=None,
                success=False,
                error=str(e),
                latency_ms=latency_ms
            )
    
    async def execute_stream(
        self,
        input_data: Union[AgentInput, Dict[str, Any], str],
        capability_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute agent with streaming output.
        
        Args:
            input_data: Input data
            capability_id: Optional capability to invoke
            skill_id: Optional skill to invoke
            context: Optional execution context
            
        Yields:
            Streaming chunks
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(input_data, str):
            agent_input = AgentInput(
                content=input_data,
                content_type=ContentType.TEXT,
                context=context,
                capability_id=capability_id,
                skill_id=skill_id,
            )
        elif isinstance(input_data, dict):
            agent_input = AgentInput(
                content=input_data,
                content_type=ContentType.JSON_OBJECT,
                context=context,
                capability_id=capability_id,
                skill_id=skill_id,
            )
        else:
            agent_input = input_data
        
        self._state = AgentState.EXECUTING
        
        try:
            input_result = await self._guardrails_engine.check_input(agent_input)
            if not input_result.passed:
                yield {
                    "type": "error",
                    "error": f"Input guardrail failed: {input_result.reason}"
                }
                return
            
            async for chunk in self._execution_engine.execute_stream(agent_input):
                yield chunk
            
            self._state = AgentState.COMPLETED
            
        except Exception as e:
            self._state = AgentState.ERROR
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def get_capability(self, capability_id: str) -> Optional[CapabilityDefinition]:
        """Get capability by ID."""
        for cap in self._config.capabilities:
            if cap.id == capability_id:
                return cap
        return None
    
    def get_skill(self, capability_id: str, skill_id: str) -> Optional[SkillDefinition]:
        """Get skill by capability and skill ID."""
        capability = self.get_capability(capability_id)
        if capability:
            for skill in capability.skills:
                if skill.id == skill_id:
                    return skill
        return None
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool by ID."""
        for tool in self._config.tools:
            if tool.tool_id == tool_id:
                return tool
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "status": "healthy" if self._state != AgentState.ERROR else "unhealthy",
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self._state.value,
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if self._llm_client:
            try:
                llm_health = await self._llm_client.health_check()
                health["llm"] = llm_health
            except Exception as e:
                health["llm"] = {"status": "unhealthy", "error": str(e)}
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown agent and cleanup resources."""
        logger.info(f"🛑 Shutting down agent: {self.name}")
        
        if self._memory_manager:
            await self._memory_manager.close()
        
        if self._llm_client:
            await self._llm_client.close()
        
        self._initialized = False
        self._state = AgentState.IDLE
        
        logger.info(f"✅ Agent shutdown complete: {self.name}")
    
    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.name}, pattern={self.execution_pattern.value})"


async def create_agent(
    config_path: Optional[Path] = None,
    config: Optional[AgentConfiguration] = None,
    runtime_config: Optional[AgentConfig] = None,
    auto_initialize: bool = True,
) -> Agent:
    """
    Factory function to create and optionally initialize an agent.
    
    Args:
        config_path: Path to PHTN-AGENT.json
        config: Pre-loaded configuration
        runtime_config: Runtime configuration
        auto_initialize: Whether to initialize automatically
        
    Returns:
        Initialized Agent instance
    """
    agent = Agent(
        config=config,
        config_path=config_path,
        runtime_config=runtime_config,
    )
    
    if auto_initialize:
        await agent.initialize()
    
    return agent
