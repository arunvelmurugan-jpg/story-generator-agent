"""
Plan and Execute Pattern for PHTN.AI Sub-Agent Framework

Implements a two-phase approach: planning then execution.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, AsyncIterator, TYPE_CHECKING

from .base import BasePattern, ExecutionContext

if TYPE_CHECKING:
    from ...agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class PlanExecutePattern(BasePattern):
    """
    Plan and Execute execution pattern.
    
    This pattern:
    1. Planning Phase: Create a detailed plan to solve the task
    2. Execution Phase: Execute each step of the plan
    3. Replan if needed when steps fail
    
    Best for:
    - Complex multi-step tasks
    - Tasks requiring coordination
    - Long-running workflows
    - Tasks with dependencies between steps
    """
    
    pattern_name = "plan_execute"
    
    PLANNING_PROMPT = """You are a planning assistant. Create a detailed plan to accomplish the given task.

Available tools:
{tools}

Create a plan as a JSON array of steps. Each step should have:
- "step_number": Sequential number
- "description": What this step accomplishes
- "action": Either "tool_call" or "reasoning"
- "tool": Tool name (if action is "tool_call")
- "tool_input": Tool input parameters (if action is "tool_call")
- "depends_on": List of step numbers this step depends on (optional)

Respond with ONLY the JSON array, no other text.

Task: {task}
"""
    
    EXECUTION_PROMPT = """You are executing step {step_number} of a plan.

Original task: {task}

Current step: {step_description}

Previous results:
{previous_results}

Execute this step and provide the result.
"""
    
    REPLAN_PROMPT = """The following step failed during execution:

Step {step_number}: {step_description}
Error: {error}

Previous successful results:
{previous_results}

Please create a revised plan to complete the remaining task. Respond with ONLY the JSON array of remaining steps.
"""
    
    async def execute(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> "AgentOutput":
        """Execute Plan and Execute pattern."""
        from ...agent import AgentOutput, ContentType
        
        self.reset_trace()
        
        plan_config = self.config.plan_execute_config or {}
        replan_on_failure = plan_config.get("replan_on_failure", True)
        max_plan_steps = plan_config.get("max_plan_steps", 10)
        
        task = input_data.content
        if isinstance(task, dict):
            task = task.get("task", json.dumps(task))
        
        plan = await self._create_plan(str(task), context, max_plan_steps)
        
        self.add_trace_step(
            step_type="planning",
            input_data={"task": task},
            output_data={"plan": plan},
        )
        
        if not plan:
            return AgentOutput(
                content="Failed to create a plan for the task.",
                content_type=ContentType.TEXT,
                success=False,
                error="Planning failed",
            )
        
        results = {}
        tool_calls = []
        
        for step in plan:
            step_num = step.get("step_number", 0)
            step_desc = step.get("description", "")
            action = step.get("action", "reasoning")
            
            try:
                if action == "tool_call":
                    tool_name = step.get("tool")
                    tool_input = step.get("tool_input", {})
                    
                    result = await self.execute_tool(tool_name, tool_input)
                    step_result = result.get("result", str(result))
                    
                    tool_calls.append({
                        "step": step_num,
                        "tool": tool_name,
                        "input": tool_input,
                        "output": step_result,
                    })
                else:
                    step_result = await self._execute_reasoning_step(
                        step, str(task), results, context
                    )
                
                results[step_num] = {
                    "description": step_desc,
                    "result": step_result,
                    "success": True,
                }
                
                self.add_trace_step(
                    step_type="step_execution",
                    input_data={"step": step},
                    output_data={"result": step_result},
                )
                
            except Exception as e:
                logger.error(f"Step {step_num} failed: {e}")
                
                if replan_on_failure:
                    new_plan = await self._replan(
                        str(task), step, str(e), results, context
                    )
                    if new_plan:
                        plan = plan[:step_num] + new_plan
                        continue
                
                results[step_num] = {
                    "description": step_desc,
                    "error": str(e),
                    "success": False,
                }
        
        final_result = await self._synthesize_results(str(task), results, context)
        
        output = AgentOutput(
            content=final_result,
            content_type=ContentType.TEXT,
            success=True,
            tool_calls=tool_calls,
            metadata={
                "plan_steps": len(plan),
                "successful_steps": sum(1 for r in results.values() if r.get("success")),
                "results": results,
            },
        )
        
        return await self.post_process(output, input_data, context)
    
    async def execute_stream(
        self,
        input_data: "AgentInput",
        context: ExecutionContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute Plan and Execute pattern with streaming."""
        self.reset_trace()
        
        plan_config = self.config.plan_execute_config or {}
        max_plan_steps = plan_config.get("max_plan_steps", 10)
        
        yield {"type": "start", "pattern": self.pattern_name}
        
        task = input_data.content
        if isinstance(task, dict):
            task = task.get("task", json.dumps(task))
        
        yield {"type": "planning_start"}
        
        plan = await self._create_plan(str(task), context, max_plan_steps)
        
        yield {
            "type": "planning_complete",
            "plan": plan,
            "step_count": len(plan) if plan else 0,
        }
        
        if not plan:
            yield {"type": "error", "error": "Failed to create plan"}
            return
        
        results = {}
        
        for step in plan:
            step_num = step.get("step_number", 0)
            step_desc = step.get("description", "")
            action = step.get("action", "reasoning")
            
            yield {
                "type": "step_start",
                "step_number": step_num,
                "description": step_desc,
                "action": action,
            }
            
            try:
                if action == "tool_call":
                    tool_name = step.get("tool")
                    tool_input = step.get("tool_input", {})
                    
                    result = await self.execute_tool(tool_name, tool_input)
                    step_result = result.get("result", str(result))
                else:
                    step_result = await self._execute_reasoning_step(
                        step, str(task), results, context
                    )
                
                results[step_num] = {
                    "description": step_desc,
                    "result": step_result,
                    "success": True,
                }
                
                yield {
                    "type": "step_complete",
                    "step_number": step_num,
                    "result": step_result,
                    "success": True,
                }
                
            except Exception as e:
                results[step_num] = {
                    "description": step_desc,
                    "error": str(e),
                    "success": False,
                }
                
                yield {
                    "type": "step_complete",
                    "step_number": step_num,
                    "error": str(e),
                    "success": False,
                }
        
        yield {"type": "synthesis_start"}
        
        final_result = await self._synthesize_results(str(task), results, context)
        
        yield {
            "type": "end",
            "result": final_result,
            "successful_steps": sum(1 for r in results.values() if r.get("success")),
            "total_steps": len(plan),
        }
    
    async def _create_plan(
        self,
        task: str,
        context: ExecutionContext,
        max_steps: int,
    ) -> List[Dict[str, Any]]:
        """Create execution plan."""
        tools_desc = self._format_tools(context.tools_available)
        
        prompt = self.PLANNING_PROMPT.format(
            tools=tools_desc,
            task=task,
        )
        
        messages = [
            {"role": "system", "content": "You are a planning assistant. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.call_llm(messages)
        content = response.get("content", "")
        
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            plan = json.loads(content)
            
            if len(plan) > max_steps:
                plan = plan[:max_steps]
            
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}")
            return []
    
    async def _execute_reasoning_step(
        self,
        step: Dict[str, Any],
        task: str,
        previous_results: Dict[int, Dict[str, Any]],
        context: ExecutionContext,
    ) -> str:
        """Execute a reasoning step."""
        results_str = self._format_results(previous_results)
        
        prompt = self.EXECUTION_PROMPT.format(
            step_number=step.get("step_number"),
            task=task,
            step_description=step.get("description"),
            previous_results=results_str,
        )
        
        messages = [
            {"role": "system", "content": context.system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.call_llm(messages)
        return response.get("content", "")
    
    async def _replan(
        self,
        task: str,
        failed_step: Dict[str, Any],
        error: str,
        previous_results: Dict[int, Dict[str, Any]],
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Create a new plan after failure."""
        results_str = self._format_results(previous_results)
        
        prompt = self.REPLAN_PROMPT.format(
            step_number=failed_step.get("step_number"),
            step_description=failed_step.get("description"),
            error=error,
            previous_results=results_str,
        )
        
        messages = [
            {"role": "system", "content": "You are a planning assistant. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.call_llm(messages)
        content = response.get("content", "")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return []
    
    async def _synthesize_results(
        self,
        task: str,
        results: Dict[int, Dict[str, Any]],
        context: ExecutionContext,
    ) -> str:
        """Synthesize final result from all step results."""
        results_str = self._format_results(results)
        
        prompt = f"""Based on the following task and execution results, provide a final comprehensive answer.

Task: {task}

Execution Results:
{results_str}

Provide a clear, complete answer to the original task based on these results.
"""
        
        messages = [
            {"role": "system", "content": context.system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.call_llm(messages)
        return response.get("content", "")
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for planning prompt."""
        if not tools:
            return "No tools available."
        
        lines = []
        for tool in tools:
            name = tool.get("name", tool.get("tool_id", "unknown"))
            desc = tool.get("description", "No description")
            lines.append(f"- {name}: {desc}")
        
        return "\n".join(lines)
    
    def _format_results(self, results: Dict[int, Dict[str, Any]]) -> str:
        """Format previous results for prompts."""
        if not results:
            return "No previous results."
        
        lines = []
        for step_num, result in sorted(results.items()):
            desc = result.get("description", "")
            if result.get("success"):
                lines.append(f"Step {step_num} ({desc}): {result.get('result', 'Completed')}")
            else:
                lines.append(f"Step {step_num} ({desc}): FAILED - {result.get('error', 'Unknown error')}")
        
        return "\n".join(lines)
