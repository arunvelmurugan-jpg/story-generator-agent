"""
API Routes for PHTN.AI Sub-Agent Framework

REST endpoints for agent operations.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["agents"])

logger = logging.getLogger(__name__)


_agents: Dict[str, Any] = {}


class ExecuteRequest(BaseModel):
    """Agent execution request."""
    content: Any = Field(..., description="Input content")
    content_type: str = Field(default="TEXT", description="Content type")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Execution context")
    stream: bool = Field(default=False, description="Enable streaming response")


class ExecuteResponse(BaseModel):
    """Agent execution response."""
    request_id: str
    agent_id: str
    status: str
    content: Any
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Agent information."""
    agent_id: str
    name: str
    version: str
    description: str
    status: str
    execution_pattern: str


@router.post("/agents/{agent_id}/execute", response_model=ExecuteResponse)
async def execute_agent(
    agent_id: str,
    request: ExecuteRequest,
    http_request: Request,
):
    """
    Execute an agent with the given input.
    
    Args:
        agent_id: Agent identifier
        request: Execution request
        http_request: HTTP request
        
    Returns:
        ExecuteResponse
    """
    agent = _agents.get(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    try:
        trace_header = getattr(http_request.state, "trace_header", None)
        
        from ..core.agent import AgentInput, AgentContext
        
        context = AgentContext(
            request_id=getattr(http_request.state, "request_id", ""),
            session_id=request.context.get("session_id", "") if request.context else "",
            user_id=request.context.get("user_id", "") if request.context else "",
            tenant_id=request.context.get("tenant_id", "") if request.context else "",
            metadata=request.context or {},
        )
        
        agent_input = AgentInput(
            content=request.content,
            content_type=request.content_type,
            context=context,
        )
        
        if request.stream:
            async def stream_generator():
                async for chunk in agent.execute_stream(agent_input):
                    yield f"data: {chunk}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        
        output = await agent.execute(agent_input)
        
        return ExecuteResponse(
            request_id=context.request_id,
            agent_id=agent_id,
            status="completed",
            content=output.content,
            content_type=output.content_type,
            metadata=output.metadata,
        )
        
    except Exception as e:
        logger.exception(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get agent information."""
    agent = _agents.get(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    config = agent.config
    
    return AgentInfo(
        agent_id=config.agent_id,
        name=config.name,
        version=config.version,
        description=config.description,
        status="active",
        execution_pattern=config.execution_config.pattern if config.execution_config else "simple",
    )


@router.get("/agents")
async def list_agents():
    """List all registered agents."""
    agents = []
    for agent_id, agent in _agents.items():
        config = agent.config
        agents.append({
            "agent_id": config.agent_id,
            "name": config.name,
            "version": config.version,
            "status": "active",
        })
    return {"agents": agents}


@router.post("/agents/{agent_id}/register")
async def register_agent(agent_id: str, config_path: str):
    """Register a new agent from configuration file."""
    try:
        from ..core.agent import Agent
        
        agent = Agent.from_config_file(config_path)
        _agents[agent_id] = agent
        
        return {"status": "registered", "agent_id": agent_id}
        
    except Exception as e:
        logger.exception(f"Agent registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    del _agents[agent_id]
    return {"status": "unregistered", "agent_id": agent_id}


def register_agent_instance(agent_id: str, agent: Any):
    """Register an agent instance programmatically."""
    _agents[agent_id] = agent


def get_agent_instance(agent_id: str) -> Optional[Any]:
    """Get a registered agent instance."""
    return _agents.get(agent_id)


@router.post("/test-llm")
async def test_llm(http_request: Request):
    """
    Test LLM connectivity with a simple prompt.
    
    Returns:
        LLM response with cost and token metrics
    """
    try:
        from ..shared.engines.llm_client import LLMClient
        from ..observability.otel_logging import get_logger
        
        test_logger = get_logger("test_llm")
        
        # Get agent config
        agent = getattr(http_request.app.state, "agent", None)
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        config_dict = {
            "model_config": {
                "provider": "openai",
                "primary_model": getattr(agent.config, "model", "gpt-4o"),
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 150,
                }
            },
            "llm_providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                }
            }
        }
        
        llm_client = LLMClient(config_dict)
        
        if not llm_client.enabled:
            return JSONResponse(
                content={
                    "status": "disabled",
                    "message": "LLM client is not enabled. Check OPENAI_API_KEY environment variable.",
                    "enabled": False,
                },
                status_code=503,
            )
        
        # Test with a simple prompt
        test_logger.info("Testing LLM connectivity with hello prompt")
        result = llm_client.synthesize(
            system_prompt="You are a helpful assistant. Respond in a friendly and concise manner.",
            context_text="Say hello and confirm you are working correctly. Keep it brief (1 sentence)."
        )
        
        if result:
            return {
                "status": "success",
                "message": "LLM is working correctly",
                "enabled": True,
                "model": llm_client.model,
                "provider": llm_client.provider,
                "response": result,
            }
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "LLM call returned no result",
                    "enabled": True,
                },
                status_code=500,
            )
        
    except Exception as e:
        logger.exception(f"LLM test failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")
