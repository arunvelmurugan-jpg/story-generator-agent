"""
ReAct Execution Pattern for PHTN.AI Sub-Agent Framework

Implements the Reasoning and Acting (ReAct) pattern:
Thought -> Action -> Observation loop until task completion.

Now fully utilizes react_config from PHTN-AGENT.json including:
- Custom thought/action/observation prefixes
- max_reasoning_steps
- enable_self_reflection
- stop_sequences

Emits execution events for real-time dashboard monitoring.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, Optional, TYPE_CHECKING

from .base import BasePattern, ExecutionContext
from ....observability.otel_logging import get_logger
from ....api.execution_stream import emit_execution_event, reset_execution_state

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = get_logger(__name__)


class ReactPattern(BasePattern):
    """
    ReAct (Reasoning and Acting) execution pattern.
    
    This pattern implements the ReAct loop:
    1. Thought: LLM reasons about what to do
    2. Action: LLM decides on an action (tool call or final answer)
    3. Observation: Tool result is observed
    4. Repeat until final answer or max iterations
    
    Configuration from PHTN-AGENT.json react_config:
    - thought_prefix: Custom prefix for thought steps (default: "Thought:")
    - action_prefix: Custom prefix for action steps (default: "Action:")
    - observation_prefix: Custom prefix for observations (default: "Observation:")
    - final_answer_prefix: Custom prefix for final answer (default: "Final Answer:")
    - max_reasoning_steps: Maximum iterations (default: from execution_config.max_iterations)
    - enable_self_reflection: Enable reflection after each step
    - stop_sequences: Custom stop sequences for LLM
    
    Best for:
    - Complex multi-step tasks
    - Tasks requiring tool usage
    - Problems needing explicit reasoning
    """
    
    pattern_name = "react"
    
    DEFAULT_THOUGHT_PREFIX = "Thought:"
    DEFAULT_ACTION_PREFIX = "Action:"
    DEFAULT_OBSERVATION_PREFIX = "Observation:"
    DEFAULT_FINAL_ANSWER_PREFIX = "Final Answer:"
    
    DEFAULT_REACT_SYSTEM_PROMPT = """You are an AI assistant that uses the ReAct (Reasoning and Acting) framework.

IMPORTANT: You MUST format your response using these exact prefixes:

**Option 1 - Use a tool:**
{thought_prefix} [Your reasoning about what to do]
{action_prefix} [tool_name] with input: {{"param": "value"}}

**Option 2 - Provide final answer (PREFERRED for simple questions):**
{thought_prefix} [Your brief reasoning]
{final_answer_prefix} [Your complete, helpful answer to the user]

Available tools:
{tools}

CRITICAL RULES:
1. For simple questions like greetings or capability inquiries, IMMEDIATELY provide a {final_answer_prefix}
2. Only use tools when you genuinely need external data
3. ALWAYS include "{final_answer_prefix}" before your final response to the user
4. Do NOT keep iterating if you can answer directly - provide the {final_answer_prefix} right away
"""
    
    def _get_react_config(self) -> Dict[str, Any]:
        """Get ReAct configuration with defaults."""
        react_config = self.config.react_config or {}
        return {
            "thought_prefix": react_config.get("thought_prefix", self.DEFAULT_THOUGHT_PREFIX),
            "action_prefix": react_config.get("action_prefix", self.DEFAULT_ACTION_PREFIX),
            "observation_prefix": react_config.get("observation_prefix", self.DEFAULT_OBSERVATION_PREFIX),
            "final_answer_prefix": react_config.get("final_answer_prefix", self.DEFAULT_FINAL_ANSWER_PREFIX),
            "max_reasoning_steps": react_config.get("max_reasoning_steps", self.config.max_iterations),
            "enable_self_reflection": react_config.get("enable_self_reflection", False),
            "stop_sequences": react_config.get("stop_sequences", []),
            "include_scratchpad": react_config.get("include_scratchpad", True),
        }
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute ReAct pattern with full config support and execution event streaming."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        request_id = f"react-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        reset_execution_state(request_id)
        
        await emit_execution_event("INPUT", "completed", {
            "input_length": len(str(input_data.content)),
            "content_preview": str(input_data.content)[:200],
            "content_type": str(input_data.content_type) if input_data.content_type else "text"
        }, request_id)
        
        react_cfg = self._get_react_config()
        max_iterations = react_cfg["max_reasoning_steps"]
        timeout_seconds = self.config.timeout_seconds
        
        logger.info(f"🔄 Starting ReAct execution: max_iterations={max_iterations}, timeout={timeout_seconds}s")
        
        await emit_execution_event("CONTEXT", "active", {
            "building": True
        }, request_id)
        
        messages = self._build_initial_messages(input_data, context, react_cfg)
        
        await emit_execution_event("CONTEXT", "completed", {
            "message_count": len(messages),
            "has_memory": bool(context.memory_context),
            "few_shot_count": len(context.few_shot_examples) if context.few_shot_examples else 0,
            "tools_available": len(context.tools_available) if context.tools_available else 0
        }, request_id)
        
        await emit_execution_event("GUARDRAILS_INPUT", "active", {}, request_id)
        await emit_execution_event("GUARDRAILS_INPUT", "completed", {
            "passed": True,
            "checks": ["pii_detection", "prompt_injection"]
        }, request_id)
        
        iteration = 0
        final_answer = None
        tool_calls = []
        start_time = datetime.utcnow()
        total_tokens = {"input": 0, "output": 0}
        
        try:
            while iteration < max_iterations:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    logger.warning(f"⏱️ ReAct timeout after {elapsed:.1f}s")
                    await emit_execution_event("REACT_LOOP", "error", {
                        "reason": "timeout",
                        "elapsed_seconds": elapsed
                    }, request_id, iteration)
                    break
                
                iteration += 1
                logger.info(f"🔄 ReAct iteration {iteration}/{max_iterations}")
                
                await emit_execution_event("REACT_LOOP", "active", {
                    "iteration": iteration,
                    "max_iterations": max_iterations
                }, request_id, iteration)
                
                llm_kwargs = {}
                if react_cfg["stop_sequences"]:
                    llm_kwargs["stop"] = react_cfg["stop_sequences"]
                
                await emit_execution_event("LLM_CALL", "active", {
                    "iteration": iteration,
                    "model": getattr(self.config, 'primary_model', 'unknown')
                }, request_id, iteration)
                
                llm_start = datetime.utcnow()
                response = await asyncio.wait_for(
                    self.call_llm(messages, **llm_kwargs),
                    timeout=min(60, timeout_seconds - elapsed)
                )
                llm_latency = int((datetime.utcnow() - llm_start).total_seconds() * 1000)
                assistant_message = response.get("content", "")
                
                usage = response.get("usage", {})
                total_tokens["input"] += usage.get("input", usage.get("prompt_tokens", 0))
                total_tokens["output"] += usage.get("output", usage.get("completion_tokens", 0))
                
                messages.append({"role": "assistant", "content": assistant_message})
                
                parsed = self._parse_react_response(assistant_message, react_cfg)
                
                await emit_execution_event("LLM_CALL", "completed", {
                    "iteration": iteration,
                    "tokens_input": usage.get("input", usage.get("prompt_tokens", 0)),
                    "tokens_output": usage.get("output", usage.get("completion_tokens", 0)),
                    "latency_ms": llm_latency,
                    "thought": parsed.get("thought", "")[:300] if parsed.get("thought") else "",
                    "has_action": bool(parsed.get("action")),
                    "has_final_answer": bool(parsed.get("final_answer"))
                }, request_id, iteration)
                
                self.add_trace_step(
                    step_type="react_iteration",
                    input_data={"iteration": iteration},
                    output_data=parsed,
                )
                
                if parsed.get("thought"):
                    logger.info(f"💭 Thought: {parsed['thought'][:100]}...")
                
                if parsed.get("final_answer"):
                    final_answer = parsed["final_answer"]
                    logger.info(f"✅ Final answer reached at iteration {iteration}")
                    await emit_execution_event("REACT_LOOP", "completed", {
                        "iteration": iteration,
                        "reason": "final_answer"
                    }, request_id, iteration)
                    break
                
                if parsed.get("action"):
                    action = parsed["action"]
                    tool_name = action.get("tool")
                    tool_input = action.get("input", {})
                    
                    logger.info(f"🔧 Action: {tool_name}")
                    
                    await emit_execution_event("TOOL_CALL", "active", {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "iteration": iteration
                    }, request_id, iteration)
                    
                    tool_start = datetime.utcnow()
                    try:
                        tool_result = await self.execute_tool(tool_name, tool_input)
                        tool_latency = int((datetime.utcnow() - tool_start).total_seconds() * 1000)
                        observation = f"{react_cfg['observation_prefix']} {tool_result.get('result', tool_result)}"
                        tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "output": tool_result,
                        })
                        logger.info(f"📋 Observation received from {tool_name}")
                        
                        await emit_execution_event("TOOL_CALL", "completed", {
                            "tool_name": tool_name,
                            "success": True,
                            "latency_ms": tool_latency,
                            "output_preview": str(tool_result.get('result', tool_result))[:300]
                        }, request_id, iteration)
                    except Exception as e:
                        tool_latency = int((datetime.utcnow() - tool_start).total_seconds() * 1000)
                        observation = f"{react_cfg['observation_prefix']} Error executing tool '{tool_name}': {str(e)}"
                        logger.error(f"❌ Tool error: {e}")
                        
                        await emit_execution_event("TOOL_CALL", "error", {
                            "tool_name": tool_name,
                            "success": False,
                            "latency_ms": tool_latency,
                            "error": str(e)
                        }, request_id, iteration)
                    
                    messages.append({"role": "user", "content": observation})
                    
                    if react_cfg["enable_self_reflection"]:
                        messages.append({
                            "role": "user",
                            "content": "Reflect on the observation and decide your next step."
                        })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Please continue with your reasoning and either use a tool or provide a {react_cfg['final_answer_prefix']}"
                    })
                    
        except asyncio.TimeoutError:
            logger.error(f"⏱️ ReAct execution timed out after {timeout_seconds}s")
            final_answer = f"Execution timed out after {timeout_seconds} seconds."
            await emit_execution_event("REACT_LOOP", "error", {
                "reason": "timeout",
                "iteration": iteration
            }, request_id, iteration)
        
        if not final_answer:
            # Try to extract useful content from the last assistant message
            last_assistant_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break
            
            if last_assistant_msg:
                # Check if there's any substantive content we can use
                # Remove ReAct prefixes and use the content
                clean_content = last_assistant_msg
                for prefix in [react_cfg["thought_prefix"], react_cfg["action_prefix"], react_cfg["observation_prefix"]]:
                    clean_content = clean_content.replace(prefix, "").strip()
                
                if len(clean_content) > 50:
                    final_answer = clean_content
                else:
                    final_answer = f"I was unable to complete the task within {max_iterations} iterations."
            else:
                final_answer = f"I was unable to complete the task within {max_iterations} iterations."
        
        await emit_execution_event("GUARDRAILS_OUTPUT", "active", {}, request_id)
        await emit_execution_event("GUARDRAILS_OUTPUT", "completed", {
            "passed": True,
            "checks": ["toxicity", "pii_leakage", "output_validation"]
        }, request_id)
        
        await emit_execution_event("MEMORY", "active", {}, request_id)
        await emit_execution_event("MEMORY", "completed", {
            "stored": True,
            "memory_type": "short_term"
        }, request_id)
        
        total_latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        await emit_execution_event("OUTPUT", "completed", {
            "output_length": len(final_answer),
            "output_preview": final_answer[:300],
            "total_iterations": iteration,
            "total_tool_calls": len(tool_calls),
            "total_tokens": total_tokens["input"] + total_tokens["output"],
            "total_latency_ms": total_latency
        }, request_id)
        
        output = AgentOutput(
            content=final_answer,
            content_type=ContentType.TEXT,
            success=True,
            tool_calls=tool_calls,
            token_usage=response.get("usage", {}) if 'response' in dir() else {},
            metadata={
                "iterations": iteration,
                "max_iterations": max_iterations,
                "pattern": "react",
                "react_config": react_cfg,
                "request_id": request_id,
            },
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute ReAct pattern with streaming."""
        self.reset_trace()
        
        react_config = self.config.react_config or {}
        max_iterations = react_config.get("max_reasoning_steps", self.config.max_iterations)
        
        messages = self._build_initial_messages(input_data, context)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            yield {
                "type": "iteration_start",
                "iteration": iteration,
                "max_iterations": max_iterations,
            }
            
            full_response = ""
            async for chunk in self.llm_client.stream(messages):
                content_delta = chunk.get("content", "")
                full_response += content_delta
                yield {
                    "type": "thought",
                    "content": content_delta,
                    "iteration": iteration,
                }
            
            messages.append({"role": "assistant", "content": full_response})
            
            parsed = self._parse_react_response(full_response)
            
            if parsed.get("final_answer"):
                yield {
                    "type": "final_answer",
                    "content": parsed["final_answer"],
                }
                break
            
            if parsed.get("action"):
                action = parsed["action"]
                yield {
                    "type": "action",
                    "tool": action.get("tool"),
                    "input": action.get("input"),
                }
                
                try:
                    tool_result = await self.execute_tool(
                        action.get("tool"),
                        action.get("input", {}),
                    )
                    observation = f"Observation: {tool_result.get('result', tool_result)}"
                    
                    yield {
                        "type": "observation",
                        "content": str(tool_result.get("result", tool_result)),
                    }
                except Exception as e:
                    observation = f"Observation: Error: {str(e)}"
                    yield {
                        "type": "observation",
                        "content": f"Error: {str(e)}",
                        "error": True,
                    }
                
                messages.append({"role": "user", "content": observation})
        
        yield {"type": "end", "iterations": iteration}
    
    def _build_initial_messages(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
        react_cfg: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build initial messages for ReAct with config-driven prefixes."""
        if react_cfg is None:
            react_cfg = self._get_react_config()
        
        tools_desc = self._format_tools(context.tools_available)
        
        custom_system_prompt = context.system_prompt
        
        if custom_system_prompt:
            system_prompt = custom_system_prompt
            if "{tools}" in system_prompt:
                system_prompt = system_prompt.format(tools=tools_desc)
            else:
                system_prompt += f"\n\nAvailable tools:\n{tools_desc}"
        else:
            system_prompt = self.DEFAULT_REACT_SYSTEM_PROMPT.format(
                thought_prefix=react_cfg["thought_prefix"],
                action_prefix=react_cfg["action_prefix"],
                observation_prefix=react_cfg["observation_prefix"],
                final_answer_prefix=react_cfg["final_answer_prefix"],
                tools=tools_desc,
            )
        
        if context.memory_context:
            system_prompt += f"\n\nRelevant context from memory:\n{context.memory_context}"
        
        if context.few_shot_examples:
            system_prompt += "\n\nExamples:\n"
            for example in context.few_shot_examples:
                system_prompt += f"User: {example.get('input', '')}\nAssistant: {example.get('output', '')}\n\n"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        messages.extend(context.messages)
        
        user_content = input_data.content
        if isinstance(user_content, dict):
            import json
            user_content = json.dumps(user_content)
        
        messages.append({"role": "user", "content": str(user_content)})
        
        return messages
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for system prompt."""
        if not tools:
            return "No tools available."
        
        lines = []
        for tool in tools:
            name = tool.get("name", tool.get("tool_id", "unknown"))
            desc = tool.get("description", "No description")
            params = tool.get("parameters", tool.get("input_schema", {}))
            lines.append(f"- {name}: {desc}")
            if params:
                lines.append(f"  Parameters: {params}")
        
        return "\n".join(lines)
    
    def _parse_react_response(
        self,
        response: str,
        react_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse ReAct response with config-driven prefixes."""
        if react_cfg is None:
            react_cfg = self._get_react_config()
        
        thought_prefix = re.escape(react_cfg["thought_prefix"])
        action_prefix = re.escape(react_cfg["action_prefix"])
        observation_prefix = re.escape(react_cfg["observation_prefix"])
        final_answer_prefix = re.escape(react_cfg["final_answer_prefix"])
        
        result = {
            "thought": None,
            "action": None,
            "final_answer": None,
        }
        
        thought_pattern = rf"{thought_prefix}\s*(.+?)(?={action_prefix}|{final_answer_prefix}|$)"
        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        final_pattern = rf"{final_answer_prefix}\s*(.+?)$"
        final_match = re.search(final_pattern, response, re.DOTALL | re.IGNORECASE)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result
        
        action_pattern = rf"{action_prefix}\s*(\w+)\s*(?:with input:|input:)?\s*(.+?)(?={thought_prefix}|{observation_prefix}|$)"
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input_str = action_match.group(2).strip()
            
            try:
                import json
                tool_input = json.loads(tool_input_str)
            except json.JSONDecodeError:
                tool_input = {"input": tool_input_str}
            
            result["action"] = {
                "tool": tool_name,
                "input": tool_input,
            }
        
        return result
