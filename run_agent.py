#!/usr/bin/env python3
"""
Run PHTN.AI Sub-Agent Framework Server — Story Generator Agent

Starts the sub-agent server using the local .phtnai/PHTN-AGENT.json configuration.
Port is configurable via AGENT_PORT (or PORT) environment variable (default 8080).

Import strategy: This script's directory is a Python package (has __init__.py).
We add the PARENT directory to sys.path so that relative imports like
`from ..observability` inside api/ and core/ resolve correctly against the
package name (the folder name: phtnai-agentic-ai).

The original /run endpoint is preserved verbatim so the UI calling
POST /run continues to work without any changes.
"""

import importlib
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
AGENT_DIR = Path(__file__).parent.resolve()      # .../Story_generator_agent/phtnai-agentic-ai
AGENT_PKG = AGENT_DIR.name                       # "phtnai-agentic-ai"
PARENT_DIR = str(AGENT_DIR.parent)               # .../Story_generator_agent

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ── Load .env FIRST so OPENAI_API_KEY is visible to all subsequent imports ─────
from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")

# ── Config-driven execution engine (reads PHTN-AGENT.json) ───────────────────
from shared.config_engine import ConfigEngine
_config_engine = ConfigEngine(config_path=AGENT_DIR / ".phtnai" / "PHTN-AGENT.json")

# ── OTEL logging (via importlib so hyphenated package name works) ─────────────
otel_mod = importlib.import_module(f"{AGENT_PKG}.observability.otel_logging")
configure_otel_logging = otel_mod.configure_otel_logging
get_logger = otel_mod.get_logger
set_trace_context = otel_mod.set_trace_context
set_trace_context_from_config = otel_mod.set_trace_context_from_config
PHtnAgentIdParts = otel_mod.PHtnAgentIdParts

configure_otel_logging(
    agent_name="story-generator-agent",
    service_name="phtnai-story-generator-framework",
    log_level="INFO",
    json_format=True,
)
logger = get_logger("phtnai.story_generator")


def main():
    """Main entry point — wires framework + /run endpoint, then starts uvicorn."""
    import uvicorn
    from pydantic import BaseModel
    from fastapi import HTTPException, APIRouter
    from fastapi.responses import JSONResponse, RedirectResponse
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

    SUBPATH = "/sub-ba-story-generate"

    # ── Load framework modules via importlib ──────────────────────────────────
    api_mod = importlib.import_module(f"{AGENT_PKG}.api")
    create_app = api_mod.create_app

    config_loader_mod = importlib.import_module(f"{AGENT_PKG}.core.config_loader")
    ConfigLoader = config_loader_mod.ConfigLoader

    agent_mod = importlib.import_module(f"{AGENT_PKG}.core.agent")
    Agent = agent_mod.Agent

    # ── Configuration ─────────────────────────────────────────────────────────
    port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", "8080")))
    config_path = AGENT_DIR / ".phtnai" / "PHTN-AGENT.json"
    phtnai_dir = config_path.parent

    logger.info("=" * 60)
    logger.info("PHTN.AI Story Generator Agent Starting")
    logger.info("=" * 60)

    logger.info(f"Loading configuration from: {config_path}")
    loader = ConfigLoader(phtnai_dir=phtnai_dir)
    config = loader.load_agent_config(config_path)
    set_trace_context_from_config(config)

    phtn_agent_id_parts = PHtnAgentIdParts.from_config(config)
    phtn_agent_id = phtn_agent_id_parts.to_header()

    set_trace_context(
        phtn_agent_id=phtn_agent_id,
        agent_id=config.agent_id,
        tenant_id=config.tenant or "",
        correlation_id="startup",
    )

    logger.info(f"Agent ID:          {config.agent_id}")
    logger.info(f"Agent Name:        {config.name}")
    logger.info(f"Version:           {config.version}")
    logger.info(f"Tenant:            {config.tenant or 'N/A'}")
    logger.info(
        f"Environment:       "
        f"{config.deployment_metadata.environment if config.deployment_metadata else 'production'}"
    )
    logger.info(
        f"Execution Pattern: "
        f"{config.execution_config.pattern.value if config.execution_config else 'SIMPLE'}"
    )
    logger.info(f"X-PHTN-Agent-ID:   {phtn_agent_id}")

    # ── Build agent + FastAPI app ─────────────────────────────────────────────
    agent = Agent(config=config)

    app = create_app(
        title=f"PHTN.AI Sub-Agent: {config.name}",
        version=config.version,
        debug=False,
    )

    # Disable default docs to serve them under subpath
    app.docs_url = None
    app.redoc_url = None

    app.state.agent = agent
    app.state.config = config
    app.state.phtn_agent_id = phtn_agent_id
    app.state.config_engine = _config_engine

    # Create router with subpath prefix
    router = APIRouter(prefix=SUBPATH)

    # ── Custom Docs Endpoints ─────────────────────────────────────────────────
    @router.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=f"{SUBPATH}/openapi.json",
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=None,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )

    @router.get("/redoc", include_in_schema=False)
    async def custom_redoc_html():
        return get_redoc_html(
            openapi_url=f"{SUBPATH}/openapi.json",
            title=app.title + " - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        )

    @router.get("/openapi.json", include_in_schema=False)
    async def custom_openapi():
        from fastapi.openapi.utils import get_openapi
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=router.routes,
        )

    @router.get("/", include_in_schema=False)
    async def root():
        """Redirect root path to docs."""
        return RedirectResponse(url=f"{SUBPATH}/docs")

    # ── /health ───────────────────────────────────────────────────────────────
    @router.get("/health")
    async def health():
        return {
            "status": "healthy",
            "agent_id": config.agent_id,
            "category": "STORY_GENERATOR",
            "port": port,
            "timestamp": time.time(),
        }

    # ── /livez  /readyz ───────────────────────────────────────────────────────
    @router.get("/livez")
    async def livez():
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz():
        return {"status": "ok"}

    # ── /.well-known/agent-card.json  (A2A discovery) ────────────────────────
    @router.get("/.well-known/agent-card.json")
    async def get_agent_card():
        endpoint_url = os.getenv("AGENT_URL", f"http://localhost:{port}")
        return {
            "id": config.agent_id,
            "name": config.name,
            "display_name": "Story Generator Agent",
            "description": config.description or (
                "Generates INVEST-compliant user stories from epics and features "
                "with acceptance criteria, governance scoring, and refinement capabilities."
            ),
            "owner": "PhotonAI",
            "endpoint_url": endpoint_url,
            "tags": ["photon", "story-generator", "user-stories", "invest", "agent-3"],
            "status": config.status.value if config.status else "Active",
            "version": config.version,
            "environment": (
                config.deployment_metadata.environment
                if config.deployment_metadata
                else os.getenv("AGENT_ENVIRONMENT", "Production")
            ),
            "runtime": "FastAPI/Uvicorn",
            "agent_type": "worker",
            "capabilities": [
                {
                    "name": "story_generation",
                    "description": "Generate user stories from epics and features",
                    "request_schema": {
                        "epics": "array — epic definitions",
                        "capabilities": "array — capability definitions",
                        "domain": "string — business domain",
                        "refine": "boolean — refine existing story",
                        "story_to_refine": "object — story to refine",
                        "user_instruction": "string — refinement instructions",
                        "grounded": "boolean — use grounded generation",
                        "fe_only": "boolean — frontend-only stories",
                        "prd_sections": "array — PRD sections for context",
                    },
                    "response_schema": {
                        "stories": "array — generated user stories",
                        "us_governance": "object — governance metadata",
                        "input_tokens": "integer",
                        "output_tokens": "integer",
                    },
                }
            ],
        }

    # ------------------------------------------------------------------
    # /run  ── original Story Generator endpoint (preserved for UI/workflow)
    #
    # Input:  {
    #   "epics": [...],
    #   "capabilities": [...],
    #   "domain": "",
    #   "refine": false,
    #   "story_to_refine": {},
    #   "user_instruction": "",
    #   "grounded": true,
    #   "fe_only": false,
    #   "prd_sections": null
    # }
    # Output: {
    #   "stories", "us_governance", "input_tokens", "output_tokens"
    # }
    # ------------------------------------------------------------------
    class StoryRequest(BaseModel):
        epics: List[Dict[str, Any]] = []
        capabilities: List[Dict[str, Any]] = []
        domain: str = ""
        refine: bool = False
        story_to_refine: Dict[str, Any] = {}
        user_instruction: str = ""
        grounded: bool = True
        fe_only: bool = False
        prd_sections: Optional[List[Dict[str, Any]]] = None

    class StoryResponse(BaseModel):
        stories: List[Dict[str, Any]] = []
        us_governance: Optional[Dict[str, Any]] = None
        input_tokens: int = 0
        output_tokens: int = 0

    @router.post("/run", response_model=StoryResponse, tags=["agent"])
    async def run_agent(req: StoryRequest):
        """Generate INVEST-compliant user stories from epics/features.

        Preserves the original /run contract so the upstream workflow
        continues to work without any changes.
        """
        t0 = time.time()
        mode = "refine" if req.refine else "generate"
        logger.info(
            f"[/run] mode={mode} epics={len(req.epics)} domain={req.domain} "
            f"grounded={req.grounded} fe_only={req.fe_only}"
        )
        try:
            input_data = {
                "epics": req.epics,
                "capabilities": req.capabilities,
                "domain": req.domain,
                "refine": req.refine,
                "story_to_refine": req.story_to_refine,
                "user_instruction": req.user_instruction,
                "grounded": req.grounded,
                "fe_only": req.fe_only,
                "prd_sections": req.prd_sections,
            }
            result, _ = await _config_engine.execute(input_data, {})

            elapsed = round(time.time() - t0, 2)
            logger.info(
                f"[/run] done "
                f"stories={len(result.get('stories', []))} "
                f"us_governance={'yes' if result.get('us_governance') else 'none'} "
                f"tokens_in={result.get('input_tokens', 0)} "
                f"tokens_out={result.get('output_tokens', 0)} "
                f"elapsed={elapsed}s"
            )
            return StoryResponse(
                stories=result.get("stories", []),
                us_governance=result.get("us_governance"),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
            )
        except Exception as e:
            logger.error(f"Story Generator error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── /agent/info endpoint ──────────────────────────────────────────────────
    @router.get("/agent/info", tags=["framework"])
    async def agent_info():
        """Return agent metadata from PHTN-AGENT.json."""
        return {
            "agent_id": config.agent_id,
            "name": config.name,
            "version": config.version,
            "category": "STORY_GENERATOR",
            "status": "running",
        }

    # ── Include router in app ─────────────────────────────────────────────────
    app.include_router(router)

    # ── Start uvicorn ─────────────────────────────────────────────────────────
    logger.info(f"Starting uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
