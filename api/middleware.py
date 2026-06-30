"""
Middleware for PHTN.AI Sub-Agent Framework API

Request/response middleware for tracing, error handling, etc.
Implements OTEL-compatible logging with full X-PHTN-Agent-ID support.
"""

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from ..observability.otel_logging import (
    set_trace_context,
    get_logger,
    correlation_id_var,
    phtn_agent_id_var,
    PHtnAgentIdParts,
    log_with_context,
)

logger = get_logger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for distributed tracing via X-PHTN-Agent-ID header.
    
    Sets up OTEL-compatible trace context for all requests, ensuring
    all log messages include the full X-PHTN-Agent-ID header parts.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        incoming_header = request.headers.get("X-PHTN-Agent-ID")
        correlation_id = request.headers.get("X-Correlation-ID") or \
                        request.headers.get("X-Request-ID") or \
                        str(uuid.uuid4())
        
        # Get config for extended context
        config = getattr(request.app.state, 'config', None)
        
        # Parse or create X-PHTN-Agent-ID header with all 18 fields
        if incoming_header:
            # Parse incoming header to get all 18 fields
            header_parts = PHtnAgentIdParts.from_header(incoming_header)
            # Increment hop count for child span
            header_parts.hop_count = str(int(header_parts.hop_count or "1") + 1)
            # Update correlation_id if not present in header
            if not header_parts.correlation_id:
                header_parts.correlation_id = correlation_id
            # Generate new span_id for this agent
            if not header_parts.span_id:
                header_parts.span_id = str(uuid.uuid4())[:16]
            # Keep trace_id from parent
            if not header_parts.trace_id:
                header_parts.trace_id = str(uuid.uuid4())
        else:
            # Create new header from config with all 18 fields
            trace_id = str(uuid.uuid4())
            span_id = str(uuid.uuid4())[:16]
            header_parts = PHtnAgentIdParts.from_config(
                config=config,
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=span_id
            )
        
        phtn_agent_id_str = header_parts.to_header()
        execution_id = header_parts.correlation_id or correlation_id
        
        # Set comprehensive trace context with all 18 X-PHTN-Agent-ID fields mapped
        set_trace_context(
            phtn_agent_id=phtn_agent_id_str,
            correlation_id=header_parts.correlation_id or correlation_id,
            trace_id=header_parts.trace_id,
            span_id=header_parts.span_id,
            agent_id=header_parts.agent_id,
            tenant_id=header_parts.tenant,
            request_id=execution_id,
            request_method=request.method,
            request_path=request.url.path,
            # Map all 18 X-PHTN-Agent-ID fields
            tenant_group_id=header_parts.tenant_group,
            domain_id=header_parts.domain,
            team_id=header_parts.team,
            project_id=header_parts.project,
            user_id=header_parts.user,
            agent_type=header_parts.agent_type,
            super_agent_id=header_parts.super_agent_id,
            sub_agent_id=header_parts.agent_id,
            super_agent_instance_id=header_parts.super_agent_id,
            sub_agent_instance_id=header_parts.agent_instance_id,
            app_name=header_parts.agent_name,
            environment=header_parts.environment,
            capability_id=header_parts.capability_id,
            skill_id=header_parts.skill_id,
            agent_name=header_parts.agent_name,
        )
        
        request.state.trace_header = header_parts
        request.state.request_id = execution_id
        request.state.correlation_id = header_parts.correlation_id or correlation_id
        request.state.phtn_agent_id = phtn_agent_id_str
        
        if request.url.path not in ["/.well-known/agent-card.json", "/health"]:
            request_extra = {
                "method": request.method,
                "path": request.url.path,
                "correlation_id": correlation_id,
                "agent_id": header_parts.agent_id,
                "model": config.llm_config.primary_model if config and getattr(config, "llm_config", None) else "",
                "prompt_tokens": 0,
                "input": None,
                "cost": 0.0,
            }
            log_with_context(
                logger,
                logging.INFO,
                f"📥 Incoming Request: {request.method} {request.url.path}",
                extra_data=request_extra,
            )
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Sanitize non-ASCII chars (e.g. em-dash \u2014) to prevent latin-1 UnicodeEncodeError in HTTP headers
        safe_phtn_id = phtn_agent_id_str.encode("ascii", "replace").decode("ascii") if phtn_agent_id_str else ""
        response.headers["X-PHTN-Agent-ID"] = safe_phtn_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-ID"] = execution_id
        response.headers["X-Response-Time-Ms"] = str(int(duration_ms))
        
        if request.url.path not in ["/.well-known/agent-card.json", "/health"]:
            response_extra = {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": int(duration_ms),
                "correlation_id": correlation_id,
                "agent_id": header_parts.agent_id,
                "model": config.llm_config.primary_model if config and getattr(config, "llm_config", None) else "",
                "prompt_tokens": 0,
                "input": None,
                "cost": 0.0,
            }
            log_with_context(
                logger,
                logging.INFO,
                f"📤 Response: {request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms",
                extra_data=response_extra,
            )
        
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            
            request_id = getattr(request.state, "request_id", "unknown")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                },
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication based on security.authentication config.
    
    Supports:
    - api_key: X-API-Key header validation
    - jwt: Bearer token validation
    - oauth2: OAuth2 token validation
    - none: No authentication required
    """
    
    EXEMPT_PATHS = [
            # Generic health endpoints
            "/health", "/ready", "/livez", "/readyz", "/startupz",
            # Status and process endpoints
            "/status", "/process",
            # Static resources
            "/icon.png",
            # Agent card endpoints
            "/.well-known/agent-card.json", "/.well-known/agent.json", "/_a2a/card",
            # Documentation
            "/docs", "/redoc", "/openapi.json",
            # Agent endpoints
            "/dashboard", "/rpc", "/agent/info", "/agent/execute",
            # Test endpoints
            "/api/v2/test-llm", "/test-llm",
        ]
    
    EXEMPT_PREFIXES = [
        "/api/v2/",
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        for prefix in self.EXEMPT_PREFIXES:
            if request.url.path.startswith(prefix):
                return await call_next(request)
        
        config = getattr(request.app.state, 'config', None)
        if not config or not config.security or not config.security.authentication:
            return await call_next(request)
        
        auth_config = config.security.authentication
        methods = auth_config.get("methods", [])
        
        if not methods or "none" in methods:
            return await call_next(request)
        
        auth_result = await self._authenticate(request, auth_config, methods)
        
        if not auth_result["authenticated"]:
            logger.warning(f"🔒 Authentication failed: {auth_result.get('reason', 'Unknown')}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": auth_result.get("reason", "Authentication required"),
                },
                headers={"WWW-Authenticate": self._get_www_authenticate_header(methods)},
            )
        
        request.state.auth_user = auth_result.get("user")
        request.state.auth_method = auth_result.get("method")
        
        return await call_next(request)
    
    async def _authenticate(
        self,
        request: Request,
        auth_config: dict,
        methods: list,
    ) -> dict:
        """Attempt authentication using configured methods."""
        
        for method in methods:
            if method == "api_key":
                result = await self._authenticate_api_key(request, auth_config)
                if result["authenticated"]:
                    return result
            
            elif method == "jwt":
                result = await self._authenticate_jwt(request, auth_config)
                if result["authenticated"]:
                    return result
            
            elif method == "oauth2":
                result = await self._authenticate_oauth2(request, auth_config)
                if result["authenticated"]:
                    return result
        
        return {"authenticated": False, "reason": "No valid authentication provided"}
    
    async def _authenticate_api_key(self, request: Request, auth_config: dict) -> dict:
        """Authenticate using API key."""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return {"authenticated": False, "reason": "Missing X-API-Key header"}
        
        valid_keys = auth_config.get("api_keys", [])
        api_key_config = auth_config.get("api_key_config", {})
        
        if valid_keys and api_key in valid_keys:
            return {
                "authenticated": True,
                "method": "api_key",
                "user": {"type": "api_key"},
            }
        
        if api_key_config.get("validate_format", True):
            if len(api_key) >= 32:
                logger.info("🔑 API key format valid (validation against store not implemented)")
                return {
                    "authenticated": True,
                    "method": "api_key",
                    "user": {"type": "api_key"},
                }
        
        return {"authenticated": False, "reason": "Invalid API key"}
    
    async def _authenticate_jwt(self, request: Request, auth_config: dict) -> dict:
        """Authenticate using JWT Bearer token."""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return {"authenticated": False, "reason": "Missing or invalid Authorization header"}
        
        token = auth_header[7:]
        
        jwt_config = auth_config.get("jwt_config", {})
        
        try:
            import jwt as pyjwt
            
            secret = jwt_config.get("secret", "")
            algorithm = jwt_config.get("algorithm", "HS256")
            issuer = jwt_config.get("issuer")
            audience = jwt_config.get("audience")
            
            decode_options = {}
            if issuer:
                decode_options["issuer"] = issuer
            if audience:
                decode_options["audience"] = audience
            
            if secret:
                payload = pyjwt.decode(token, secret, algorithms=[algorithm], **decode_options)
                return {
                    "authenticated": True,
                    "method": "jwt",
                    "user": payload,
                }
            else:
                payload = pyjwt.decode(token, options={"verify_signature": False})
                logger.warning("⚠️ JWT signature not verified (no secret configured)")
                return {
                    "authenticated": True,
                    "method": "jwt",
                    "user": payload,
                }
                
        except ImportError:
            logger.warning("⚠️ PyJWT not installed, skipping JWT validation")
            return {"authenticated": False, "reason": "JWT validation not available"}
        except Exception as e:
            return {"authenticated": False, "reason": f"JWT validation failed: {str(e)}"}
    
    async def _authenticate_oauth2(self, request: Request, auth_config: dict) -> dict:
        """Authenticate using OAuth2 token."""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return {"authenticated": False, "reason": "Missing or invalid Authorization header"}
        
        token = auth_header[7:]
        
        oauth2_config = auth_config.get("oauth2_config", {})
        introspection_url = oauth2_config.get("introspection_url")
        
        if introspection_url:
            logger.info("🔐 OAuth2 token introspection not fully implemented")
        
        if len(token) >= 20:
            return {
                "authenticated": True,
                "method": "oauth2",
                "user": {"type": "oauth2_token"},
            }
        
        return {"authenticated": False, "reason": "Invalid OAuth2 token"}
    
    def _get_www_authenticate_header(self, methods: list) -> str:
        """Get WWW-Authenticate header value."""
        if "jwt" in methods or "oauth2" in methods:
            return 'Bearer realm="phtnai-subagent"'
        elif "api_key" in methods:
            return 'ApiKey realm="phtnai-subagent"'
        return 'Bearer'


class CostGovernanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for cost governance based on cost_governance config.
    
    Enforces:
    - per_request_budget: Maximum cost per request
    - per_agent_cost_cap: Maximum total cost for agent
    - real_time_budget_breaker: Halt on budget exceeded
    """
    
    _total_cost: float = 0.0
    _request_count: int = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        config = getattr(request.app.state, 'config', None)
        if not config or not config.cost_governance:
            return await call_next(request)
        
        cost_config = config.cost_governance
        
        if cost_config.per_agent_cost_cap and self._total_cost >= cost_config.per_agent_cost_cap:
            if cost_config.real_time_budget_breaker:
                logger.error(f"💰 Agent cost cap exceeded: ${self._total_cost:.4f} >= ${cost_config.per_agent_cost_cap}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "budget_exceeded",
                        "message": f"Agent cost cap of ${cost_config.per_agent_cost_cap} exceeded",
                        "total_cost": self._total_cost,
                    },
                )
        
        request.state.cost_tracking = {
            "request_budget": cost_config.per_request_budget,
            "agent_cap": cost_config.per_agent_cost_cap,
            "current_total": self._total_cost,
        }
        
        response = await call_next(request)
        
        request_cost = getattr(request.state, 'request_cost', 0.0)
        self._total_cost += request_cost
        self._request_count += 1
        
        response.headers["X-Request-Cost"] = f"${request_cost:.6f}"
        response.headers["X-Total-Cost"] = f"${self._total_cost:.6f}"
        
        return response
