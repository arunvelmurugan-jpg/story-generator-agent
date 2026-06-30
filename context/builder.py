"""
Context Builder for PHTN.AI Sub-Agent Framework

Builds execution context from input, memory, and configuration.
Now fully utilizes prompt_config from PHTN-AGENT.json including:
- system_prompt injection
- few-shot examples
- prompt variables
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..observability.otel_logging import get_logger

if TYPE_CHECKING:
    from ..core.config_loader import ContextConfig, PromptConfig, AgentConfiguration
    from ..core.agent import AgentInput
    from ..core.execution.patterns.base import ExecutionContext
    from ..memory.manager import MemoryManager

logger = get_logger(__name__)


class ContextBuilder:
    """
    Builds execution context for agent patterns.
    
    Features:
    - System prompt injection from prompt_config
    - Memory context integration
    - RAG document retrieval
    - Few-shot example selection from config
    - Token budget management
    - Prompt variable substitution
    """
    
    def __init__(
        self,
        config: Optional["ContextConfig"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        prompt_config: Optional["PromptConfig"] = None,
        agent_config: Optional["AgentConfiguration"] = None,
    ):
        """
        Initialize ContextBuilder.
        
        Args:
            config: Context configuration
            memory_manager: Memory manager instance
            prompt_config: Prompt configuration with system_prompt
            agent_config: Full agent configuration
        """
        self.config = config
        self.memory_manager = memory_manager
        self.prompt_config = prompt_config
        self.agent_config = agent_config
        
        logger.debug("ContextBuilder initialized")
    
    async def build(self, input_data: "AgentInput") -> "ExecutionContext":
        """
        Build execution context from input.
        
        Args:
            input_data: Agent input
            
        Returns:
            ExecutionContext with system_prompt, few_shot_examples, etc.
        """
        from ..core.execution.patterns.base import ExecutionContext
        
        context = ExecutionContext()
        
        context.system_prompt = self._build_system_prompt(input_data)
        
        context.few_shot_examples = self._get_few_shot_examples()
        
        context.prompt_variables = self._get_prompt_variables(input_data)
        
        if self.memory_manager:
            memory_context = await self.memory_manager.get_context(
                session_id=input_data.context.request_id,
            )
            context.memory_context = memory_context
        
        if self.config:
            rag_config = getattr(self.config, 'retrieval_augmentation', None)
            if rag_config and getattr(rag_config, 'enabled', False):
                docs = await self._retrieve_documents(input_data)
                context.retrieved_documents = docs
        
        return context
    
    def _build_system_prompt(self, input_data: "AgentInput") -> Optional[str]:
        """Build system prompt from config with variable substitution."""
        if not self.prompt_config:
            return None
        
        system_prompt = self.prompt_config.system_prompt
        if not system_prompt:
            return None
        
        variables = self._get_prompt_variables(input_data)
        
        try:
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                if placeholder in system_prompt:
                    system_prompt = system_prompt.replace(placeholder, str(var_value))
        except Exception as e:
            logger.warning(f"Error substituting prompt variables: {e}")
        
        return system_prompt
    
    def _get_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Get few-shot examples from prompt config."""
        if not self.prompt_config:
            return []
        
        examples = self.prompt_config.examples or []
        
        if self.config:
            few_shot_config = getattr(self.config, 'few_shot_learning', None)
            if few_shot_config:
                max_examples = few_shot_config.get("max_examples", 3) if isinstance(few_shot_config, dict) else getattr(few_shot_config, 'max_examples', 3)
                examples = examples[:max_examples]
        
        return examples
    
    def _get_prompt_variables(self, input_data: "AgentInput") -> Dict[str, Any]:
        """Get prompt variables for substitution."""
        variables = {}
        
        if self.prompt_config and self.prompt_config.variables:
            for var_name in self.prompt_config.variables:
                variables[var_name] = ""
        
        if self.agent_config:
            variables["agent_name"] = self.agent_config.name
            variables["agent_id"] = self.agent_config.agent_id
            variables["agent_description"] = self.agent_config.description or ""
            variables["agent_version"] = self.agent_config.version
            
            if self.agent_config.domain:
                variables["domain_industry"] = self.agent_config.domain.industry or ""
                variables["domain_subdomain"] = self.agent_config.domain.sub_domain or ""
            
            if self.agent_config.tools:
                tool_names = [t.name for t in self.agent_config.tools]
                variables["available_tools"] = ", ".join(tool_names)
        
        if input_data.context:
            variables["user_id"] = input_data.context.user_id or ""
            variables["tenant_id"] = input_data.context.tenant_id or ""
            variables["environment"] = input_data.context.environment or ""
        
        if input_data.parameters:
            variables.update(input_data.parameters)
        
        return variables
    
    async def _retrieve_documents(
        self,
        input_data: "AgentInput",
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for RAG."""
        if not self.memory_manager:
            return []
        
        query = input_data.content
        if isinstance(query, dict):
            query = query.get("query", str(query))
        
        rag_config = getattr(self.config, 'retrieval_augmentation', {})
        top_k = rag_config.get("top_k", 5) if isinstance(rag_config, dict) else getattr(rag_config, 'top_k', 5)
        
        try:
            docs = await self.memory_manager.semantic_search(
                query=str(query),
                top_k=top_k,
            )
            return docs
        except Exception as e:
            logger.warning(f"Document retrieval failed: {e}")
            return []
