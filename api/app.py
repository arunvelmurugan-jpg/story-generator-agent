"""
FastAPI Application for PHTN.AI Sub-Agent Framework

Main application factory and configuration.
Implements OTEL-compatible logging with X-PHTN-Agent-ID support.
Includes real-time execution monitoring dashboard.
"""

import os
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes import router
from .jsonrpc import router as jsonrpc_router
from .execution_stream import router as execution_stream_router
from .middleware import TracingMiddleware, ErrorHandlerMiddleware, AuthenticationMiddleware, CostGovernanceMiddleware
from .health import HealthChecker, HealthCheckConfig
from ..observability.otel_logging import get_logger

logger = get_logger(__name__)

health_checker: Optional[HealthChecker] = None

STATIC_DIR = Path(__file__).parent.parent / "static"


def generate_agent_card(app: FastAPI, request: Optional[Request] = None) -> Dict[str, Any]:
    """Generate the A2A v1.0 agent card.

    Card shape matches the PhotonAI reference layout exactly so the
    downstream database mapper + UI can ingest agent cards with a single
    uniform schema:
      - supportedInterfaces: rpc / agent-card / health / status
      - provider, iconUrl (no /static/ prefix), version, documentationUrl
      - capabilities (4 boolean flags)
      - securitySchemes: {} ALWAYS (security info lives elsewhere)
      - security: [] ALWAYS
      - defaultInputModes / defaultOutputModes
      - skills[]: id, name, description, tags (snake_case),
        examples, inputModes, outputModes, inputSchema, outputSchema
      - signatures: []
    Per-skill input/output schemas are sourced from the agent's own
    prompt_config schemas in PHTN-AGENT.json (no schema invention).
    NO extensions.phtnai block — uniform shape across all agents.
    """
    config = getattr(app.state, "config", None)

    base_url = "http://localhost:8000"
    if request is not None:
        try:
            base_url = str(request.base_url).rstrip("/")
        except Exception:
            pass

    PUBLIC_HOST = os.getenv("PHTNAI_PUBLIC_HOST") or "https://az-dev.phtn.ai"
    agent_id = getattr(config, "agent_id", "") if config else ""
    slug = os.getenv("PHTNAI_AGENT_SLUG") or ""
    public_base = f"{PUBLIC_HOST}/{slug}" if slug else base_url

    # Empty-config fallback
    if config is None:
        return {
            "name": app.title,
            "description": "PHTN.AI Sub-Agent",
            "supportedInterfaces": [
                {"url": f"{base_url}/rpc", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
                {"url": f"{base_url}/.well-known/agent-card.json", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
                {"url": f"{base_url}/health", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
                {"url": f"{base_url}/status", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            ],
            "provider": {"organization": "PhotonAI", "url": "https://phtnai.com"},
            "iconUrl": f"{base_url}/icon.png",
            "version": app.version,
            "documentationUrl": f"{base_url}/docs",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
                "extendedAgentCard": False,
            },
            "securitySchemes": {},
            "security": [],
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"],
            "skills": [],
            "signatures": [],
        }

    # ---- per-skill input/output schemas from agent-level prompt_config ----
    pc_in = (getattr(config, "prompt_config", None).input_schema or {}) if (config.prompt_config and getattr(config.prompt_config, "input_schema", None)) else {}
    pc_out = (getattr(config, "prompt_config", None).output_schema or {}) if (config.prompt_config and getattr(config.prompt_config, "output_schema", None)) else {}

    in_validation = pc_in.get("validation", {}) if isinstance(pc_in, dict) else {}
    in_required = list(in_validation.get("requiredFields") or [])
    in_props = in_validation.get("properties") or {}
    out_props = (pc_out.get("properties") or {}) if isinstance(pc_out, dict) else {}

    agent_input_schema = {
        "type": "object",
        "required": in_required,
        "properties": in_props,
    }
    agent_output_schema = {
        "type": "object",
        "properties": out_props,
    }

    # ---- skills (snake_case tags) ----
    def _snake(s: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", (s or "").lower()).strip("_")

    category_tag = _snake(config.category) if config.category else "general"
    skills_list: List[Dict[str, Any]] = []
    for skill in (config.skills or []):
        sid = skill.get("skill_id") or skill.get("id") or ""
        sname = skill.get("name") or sid or "Skill"
        sdesc = skill.get("description") or f"Skill: {sname}"

        # Tags: full snake-cased skill_id + each part + category, deduped
        sid_snake = _snake(sid)
        parts = [_snake(p) for p in re.split(r"[._\-]", sid) if p]
        tags = list(dict.fromkeys([t for t in ([sid_snake] + parts + [category_tag]) if t]))

        examples = [
            f"Run {sname} for the current request.",
            "{\"input\": \"" + f"Invoke {sid} with request payload" + "\", \"session_id\": \"" + sid + "-1\"}",
        ]

        scfg = skill.get("config") or {}
        skill_input_schema = scfg.get("input_schema") or agent_input_schema
        skill_output_schema = scfg.get("output_schema") or agent_output_schema

        skills_list.append({
            "id": sid,
            "name": sname,
            "description": sdesc,
            "tags": tags,
            "examples": examples,
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json", "text/plain"],
            "inputSchema": skill_input_schema,
            "outputSchema": skill_output_schema,
        })

    for tool in (config.tools or []):
        tool_tags = list(dict.fromkeys([
            _snake(getattr(tool, "tool_id", "")),
            _snake(getattr(tool, "type", "") or "tool"),
            category_tag,
        ]))
        skills_list.append({
            "id": tool.tool_id,
            "name": tool.name,
            "description": tool.description or f"Tool: {tool.name}",
            "tags": [t for t in tool_tags if t],
            "examples": [],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
            "inputSchema": agent_input_schema,
            "outputSchema": agent_output_schema,
        })

    if not skills_list:
        skills_list.append({
            "id": "default-skill",
            "name": "General Assistance",
            "description": config.description or "General agent assistance",
            "tags": ["general", "assistant", category_tag],
            "examples": ["How can you help me?"],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"],
            "inputSchema": agent_input_schema,
            "outputSchema": agent_output_schema,
        })

    streaming_enabled = False
    try:
        streaming_enabled = bool(config.execution_config.enable_streaming) if config.execution_config else False
    except Exception:
        streaming_enabled = False

    # ---- top-level card (NO extensions block) ----
    return {
        "name": config.name,
        "description": config.description or f"PHTN.AI Agent: {config.name}",
        "supportedInterfaces": [
            {"url": f"{public_base}/rpc", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": f"{public_base}/.well-known/agent-card.json", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": f"{public_base}/health", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": f"{public_base}/status", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
        ],
        "provider": {
            "organization": config.owner or "PhotonAI",
            "url": "https://phtnai.com",
        },
        "iconUrl": f"{public_base}/icon.png",
        "version": config.version,
        "documentationUrl": f"{public_base}/docs",
        "capabilities": {
            "streaming": streaming_enabled,
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "extendedAgentCard": False,
        },
        "securitySchemes": {},
        "security": [],
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": skills_list,
        "signatures": [],
    }



def create_app(
    title: str = "PHTN.AI Sub-Agent API",
    version: str = "2.0.0",
    debug: bool = False,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        title: API title
        version: API version
        debug: Enable debug mode
        cors_origins: CORS allowed origins
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        root_path=os.getenv("PHTNAI_ROOT_PATH", ""),
        title=title,
        version=version,
        description="Enterprise-grade Sub-Agent Framework API",
        debug=debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(CostGovernanceMiddleware)
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(TracingMiddleware)
    
    app.include_router(router, prefix="/api/v2")
    app.include_router(jsonrpc_router, prefix="/api/v2", tags=["jsonrpc"])
    app.include_router(execution_stream_router, prefix="/api/v2", tags=["execution"])
    
    global health_checker
    health_config = HealthCheckConfig(
        enabled=True,
        include_details=True,
        liveness_enabled=True,
        readiness_enabled=True,
        startup_enabled=True,
        check_llm=True,
        check_tools=True,
        check_memory=True,
        check_mcp=True,
    )
    health_checker = HealthChecker(config=health_config, version=version)
    app.state.health_checker = health_checker
    
    app.include_router(health_checker.create_router(), tags=["Health"])

    # ─── LLM cost telemetry (cumulative since pod start) ────────────────────
    @app.get("/api/llm-usage", tags=["Observability"])
    async def llm_usage():
        """Return cumulative LLM token usage + USD cost since pod start.

        Aggregated across every LLMRouter instance in this process. Reset on
        pod restart. Pricing source: LLMRouter._get_default_pricing().
        """
        from ..llm.router import get_global_usage
        return get_global_usage()

    
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    @app.on_event("startup")
    async def startup():
        logger.info(f"Starting {title} v{version}")
        if health_checker:
            health_checker.mark_startup_complete()
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info(f"Shutting down {title}")
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """
        Serve the Sub-Agent Execution Monitor dashboard.
        
        Provides real-time visualization of agent execution phases:
        - 75% left panel: Execution flow with Mermaid.js
        - 25% right panel: Chat interface
        """
        dashboard_path = STATIC_DIR / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path, media_type="text/html")
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Please ensure dashboard.html exists in the static folder.</p>",
            status_code=404
        )
    
    @app.get("/.well-known/agent-card.json")
    async def agent_card(request: Request):
        """
        Agent Card endpoint for A2A protocol discovery.
        
        Returns agent metadata, capabilities, skills, and tools
        in a standardized format for agent-to-agent communication.
        Compliant with A2A Protocol v1.0 specification.
        """
        return JSONResponse(
            content=generate_agent_card(app, request),
            media_type="application/json"
        )
    
    @app.get("/.well-known/agent.json")
    async def agent_json(request: Request):
        """Alias for agent-card.json for compatibility."""
        return JSONResponse(
            content=generate_agent_card(app, request),
            media_type="application/json"
        )
    

    @app.get("/status")
    async def status_endpoint(request: Request):
        """Runtime status snapshot for the agent (A2A discovery aux endpoint)."""
        config = getattr(app.state, "config", None)
        hc = getattr(app.state, "health_checker", None)
        ready = True
        try:
            if hc is not None and hasattr(hc, "is_ready"):
                ready = bool(hc.is_ready())
        except Exception:
            ready = True
        return JSONResponse(
            content={
                "agent_id": getattr(config, "agent_id", None),
                "name": getattr(config, "name", app.title),
                "version": getattr(config, "version", app.version),
                "status": (config.status.value if (config and config.status) else "unknown"),
                "ready": ready,
                "category": getattr(config, "category", None),
                "tenant": getattr(config, "tenant", None),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            media_type="application/json",
        )


    @app.post("/process")
    async def process_endpoint(request: Request):
        """A2A reference '/process' endpoint.

        Accepts JSON ({input, session_id} or full A2A {message: {parts: [...]}})
        or plain text. Wraps the body in a JSON-RPC message/send envelope and
        delegates to the existing JSON-RPC handler — single source of truth
        for processing logic.
        """
        from .jsonrpc import handle_message_send, create_success_response, create_error_response
        try:
            ctype = (request.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                body = await request.json()
            else:
                raw = (await request.body()).decode("utf-8", errors="replace")
                body = {"input": raw}
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "bad_request", "message": f"Could not parse body: {e}"})

        if isinstance(body, dict) and isinstance(body.get("message"), dict) and body["message"].get("parts"):
            params = {"message": body["message"], "contextId": body.get("contextId") or body.get("session_id")}
        else:
            text = body.get("input") if isinstance(body, dict) else str(body)
            if text is None and isinstance(body, dict):
                text = body.get("text") or body.get("message") or ""
            if not isinstance(text, str):
                import json as _json
                text = _json.dumps(text)
            params = {
                "message": {"parts": [{"kind": "text", "text": text}]},
                "contextId": (body.get("session_id") if isinstance(body, dict) else None),
            }
        try:
            result = await handle_message_send(params, app.state)
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "process_failed", "message": str(e)})

    @app.get("/icon.png")
    async def icon_endpoint():
        """Serve a 1x1 transparent PNG so the iconUrl in the card resolves."""
        import base64
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        return Response(content=png, media_type="image/png")

    @app.get("/status")
    async def status_endpoint(request: Request):
        """Runtime status snapshot (A2A discovery aux endpoint)."""
        config = getattr(app.state, "config", None)
        hc = getattr(app.state, "health_checker", None)
        ready = True
        try:
            if hc is not None and hasattr(hc, "is_ready"):
                ready = bool(hc.is_ready())
        except Exception:
            ready = True
        return JSONResponse(
            content={
                "agent_id": getattr(config, "agent_id", None),
                "name": getattr(config, "name", app.title),
                "version": getattr(config, "version", app.version),
                "status": (config.status.value if (config and config.status) else "unknown"),
                "ready": ready,
                "category": getattr(config, "category", None),
                "tenant": getattr(config, "tenant", None),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            media_type="application/json",
        )

    return app
