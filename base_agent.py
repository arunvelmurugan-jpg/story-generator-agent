"""
Base FastAPI agent scaffold.
Every agent imports `create_agent_app` to get a pre-configured FastAPI app
with /health, /.well-known/agent-card.json, and /a2a endpoints.
"""
import os
import logging
from typing import Any, Optional, Tuple

from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def create_agent_app(
    agent_id: str,
    agent_name: str,
    description: str,
    port: int,
    version: str = "1.0.0",
    skills: Optional[list[dict]] = None,
    root_path: str = "",
) -> Tuple[FastAPI, APIRouter]:
    """
    Create a standard FastAPI application for an agent with:
      - CORS
      - /health
      - /.well-known/agent-card.json  (A2A discovery)
      - /a2a                           (JSON-RPC stub)
    Returns: (app, router) - add your endpoints to the router
    """
    # Configure docs URLs to work with subpath
    docs_url = f"{root_path}/docs" if root_path else "/docs"
    redoc_url = f"{root_path}/redoc" if root_path else "/redoc"
    openapi_url = f"{root_path}/openapi.json" if root_path else "/openapi.json"
    
    app = FastAPI(
        title=agent_name,
        description=description,
        version=version,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- A2A Agent Card (Story Generator Agent) ----------
    endpoint_url = os.getenv("AGENT_URL", f"http://localhost:{port}")
    agent_card: dict[str, Any] = {
        "id": "Story Generator Agent",
        "name": "Story Generator Agent",
        "display_name": "Story Generator Agent",
        "description": (
            "Generates INVEST-compliant user stories with acceptance criteria from epics/features; "
            "supports parallel per-epic generation, story refinement, optional US governance scoring, "
            "and PRD traceability context."
        ),
        "owner": "PhotonAI",
        "endpoint_url": endpoint_url,
        "tags": [
            "photon",
            "user-stories",
            "invest",
            "acceptance-criteria",
            "epics",
            "backlog",
            "agent-5",
        ],
        "status": "Active",
        "version": version,
        "environment": os.getenv("AGENT_ENVIRONMENT", "Development"),
        "runtime": "FastAPI/Uvicorn",
        "agent_type": "worker",
        "sla_uptime_target": "99.70%",
        "ontology": {
            "domain": "Software Quality Assurance",
            "sub_domain": "User Story Authoring & Backlog Refinement",
            "industry_standards": [
                "ISTQB",
                "Agile",
                "INVEST",
                "User story best practices",
            ],
            "knowledge_base": (
                "Epic and feature input, domain guidance, grounded vs enhanced modes, FE-only scope, "
                "story refine mode, US governance scoring with PRD sections"
            ),
            "terminology": {
                "standard": "QA & Agile Terms",
                "version": "v1.0",
            },
        },
        "schemas": "StoryGeneratorAgentv1.json",
        "llm": os.getenv("AGENT_LLM", "gpt-4o"),
        "data_access_scope": (
            "Read: epics, capabilities, domain, story refine payload, optional PRD sections; "
            "Write: user stories, acceptance criteria, governance scores, token usage"
        ),
        "capabilities": [
            {
                "name": "stories_run",
                "description": (
                    "Generate user stories from epics (parallel per epic), refine a single story from "
                    "instruction, or return governance scoring over the full story set"
                ),
                "input_schema": None,
                "output_schema": None,
                "request_schema": {
                    "epics": "array — epics with features (primary input for generation)",
                    "capabilities": "array — optional backlog context",
                    "domain": "string — business/domain context for guidance",
                    "refine": "boolean — AI edit mode for one story",
                    "story_to_refine": "object — story to refine when refine is true",
                    "user_instruction": "string — instruction for story refinement",
                    "grounded": "boolean — strict PRD/epic scope vs enhanced mode",
                    "fe_only": "boolean — front-end scope emphasis",
                    "prd_sections": "array — optional traceability context for US governance",
                },
                "response_schema": {
                    "stories": "array — generated or refined user stories",
                    "us_governance": "object — optional overallScore, passThreshold, subMetrics",
                    "input_tokens": "integer",
                    "output_tokens": "integer",
                },
            }
        ],
        "skills": {
            "primary_skills": [
                "INVEST user story generation",
                "Acceptance criteria authoring",
                "Parallel epic processing",
                "Story refinement from user instruction",
            ],
            "secondary_skills": [
                "US governance scoring",
                "Domain-aware guidance",
                "REST API",
                "LLM-assisted JSON structuring",
            ],
            "skill_level": "expert",
            "certifications": [
                "Agile backlog design",
                "Requirements traceability",
            ],
        },
        "compliance": {
            "regulations": [
                "GDPR",
                "SOC 2",
                "ISO 27001",
            ],
            "certifications": [
                "SOC 2 Type II",
                "ISO 27001",
            ],
            "audit_trail": True,
            "data_privacy": {
                "gdpr_compliant": True,
                "hipaa_compliant": False,
                "pci_dss_compliant": False,
                "ccpa_compliant": True,
            },
            "security_measures": [
                "TLS 1.3",
                "Data Encryption at Rest",
                "Role-Based Access Control",
                "API Key Authentication",
            ],
        },
        "auth_type": "none",
        "permissions": "(Agent Builder) Full edit, deploy, delete",
        "data_retention_policy": "30 days, retain until task completion",
        "pii_handling_policy": "none",
        "a2a_signature": "none",
        "interfaces": "API Approach",
        "input_channels": [
            "application/json",
            "text/plain",
        ],
        "output_channels": [
            "application/json",
            "text/plain",
        ],
        "interaction_protocols": "REST/JSON, JSON-RPC",
        "change_log": (
            "v1.0.0 - Story Generator: epics to INVEST stories; refine mode; optional US governance"
        ),
        "update_by": "phtnai",
        "approval_status": "Approved",
        "review_date": "03/26/2026",
        "transports_supported": [
            "http",
            "https",
        ],
        "protocols_supported": [
            "REST",
            "JSON-RPC",
        ],
        "classification": "Software Quality Assurance",
        "scope": "User story generation pipeline (Agent 5)",
        "maturity_level": "Production",
        "agent_name": "Story Generator Agent",
    }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": agent_id, "port": port}

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        return agent_card

    @app.post("/a2a")
    async def a2a_endpoint(request: Request):
        """JSON-RPC style A2A entry point."""
        body = await request.json()
        method = body.get("method", "")
        if method == "a2a.agent.get_card":
            return {"jsonrpc": "2.0", "id": body.get("id"), "result": agent_card}
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    router = APIRouter()


    return app, router
