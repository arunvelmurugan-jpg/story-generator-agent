"""
Health Check Module for PHTN.AI Sub-Agent Framework

Provides Kubernetes-style health probes:
- /health - General health status with component details
- /livez - Liveness probe (is the process running?)
- /readyz - Readiness probe (is the service ready for traffic?)
- /startupz - Startup probe (has the service finished starting?)

Compatible with Kubernetes probes and standard monitoring systems.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from ..observability.otel_logging import get_logger

if TYPE_CHECKING:
    from ..llm.router import LLMRouter
    from ..tools.registry import ToolRegistry
    from ..memory.manager import MemoryManager
    from ..mcp.manager import MCPManager

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentStatus(str, Enum):
    """Component status values."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: ComponentStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        if self.last_check:
            result["last_check"] = self.last_check
        return result


@dataclass
class HealthResponse:
    """Health check response."""
    status: HealthStatus
    version: str
    uptime_seconds: float
    timestamp: str
    components: List[ComponentHealth] = field(default_factory=list)
    
    def to_dict(self, include_details: bool = True) -> Dict[str, Any]:
        result = {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp,
        }
        if include_details and self.components:
            result["components"] = [c.to_dict() for c in self.components]
        return result


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    include_details: bool = True
    
    liveness_enabled: bool = True
    readiness_enabled: bool = True
    startup_enabled: bool = False
    
    check_llm: bool = True
    check_tools: bool = True
    check_memory: bool = True
    check_mcp: bool = True
    check_dependencies: bool = True
    check_disk: bool = False
    check_memory_usage: bool = False
    
    max_memory_percent: float = 90.0
    max_disk_percent: float = 90.0
    max_latency_ms: int = 5000


class HealthChecker:
    """
    Health checker for sub-agent framework.
    
    Provides comprehensive health monitoring with:
    - Component-level health checks
    - Kubernetes-compatible probe endpoints
    - Configurable check intervals and thresholds
    """
    
    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        version: str = "1.0.0",
    ):
        """
        Initialize health checker.
        
        Args:
            config: Health check configuration
            version: Application version
        """
        self.config = config or HealthCheckConfig()
        self.version = version
        self._start_time = time.time()
        self._startup_complete = False
        self._last_health_check: Optional[HealthResponse] = None
        
        self._llm_router: Optional["LLMRouter"] = None
        self._tool_registry: Optional["ToolRegistry"] = None
        self._memory_manager: Optional["MemoryManager"] = None
        self._mcp_manager: Optional["MCPManager"] = None
        
        logger.info("HealthChecker initialized")
    
    def set_components(
        self,
        llm_router: Optional["LLMRouter"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        mcp_manager: Optional["MCPManager"] = None,
    ):
        """Set component references for health checks."""
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._memory_manager = memory_manager
        self._mcp_manager = mcp_manager
    
    def mark_startup_complete(self):
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info("Startup marked as complete")
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time
    
    async def check_health(self) -> HealthResponse:
        """
        Perform comprehensive health check.
        
        Returns:
            HealthResponse with component statuses
        """
        components = []
        overall_status = HealthStatus.HEALTHY
        
        if self.config.check_llm:
            llm_health = await self._check_llm()
            components.append(llm_health)
            if llm_health.status == ComponentStatus.DOWN:
                overall_status = HealthStatus.UNHEALTHY
            elif llm_health.status == ComponentStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        if self.config.check_tools:
            tools_health = await self._check_tools()
            components.append(tools_health)
            if tools_health.status == ComponentStatus.DOWN:
                overall_status = HealthStatus.UNHEALTHY
        
        if self.config.check_memory:
            memory_health = await self._check_memory()
            components.append(memory_health)
            if memory_health.status == ComponentStatus.DOWN:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        if self.config.check_mcp:
            mcp_health = await self._check_mcp()
            components.append(mcp_health)
        
        if self.config.check_memory_usage:
            system_memory = await self._check_system_memory()
            components.append(system_memory)
            if system_memory.status == ComponentStatus.DOWN:
                overall_status = HealthStatus.UNHEALTHY
        
        if self.config.check_disk:
            disk_health = await self._check_disk()
            components.append(disk_health)
            if disk_health.status == ComponentStatus.DOWN:
                overall_status = HealthStatus.UNHEALTHY
        
        response = HealthResponse(
            status=overall_status,
            version=self.version,
            uptime_seconds=self.uptime_seconds,
            timestamp=datetime.utcnow().isoformat() + "Z",
            components=components,
        )
        
        self._last_health_check = response
        return response
    
    async def check_liveness(self) -> bool:
        """
        Liveness check - is the process alive?
        
        Returns True if the process is running and responsive.
        This should be a lightweight check.
        """
        return True
    
    async def check_readiness(self) -> bool:
        """
        Readiness check - is the service ready for traffic?
        
        Returns True if the service can handle requests.
        """
        if not self._startup_complete:
            return False
        
        if self.config.check_llm and self._llm_router:
            try:
                healthy = await self._llm_router.health_check()
                if not healthy:
                    return False
            except Exception:
                return False
        
        return True
    
    async def check_startup(self) -> bool:
        """
        Startup check - has the service finished starting?
        
        Returns True once startup is complete.
        """
        return self._startup_complete
    
    async def _check_llm(self) -> ComponentHealth:
        """Check LLM provider health."""
        start = time.time()
        
        if not self._llm_router:
            return ComponentHealth(
                name="llm",
                status=ComponentStatus.UNKNOWN,
                message="LLM router not configured",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        
        try:
            healthy = await self._llm_router.health_check()
            latency = (time.time() - start) * 1000
            
            if healthy:
                return ComponentHealth(
                    name="llm",
                    status=ComponentStatus.UP,
                    latency_ms=latency,
                    details={
                        "primary_model": getattr(self._llm_router, 'primary_model', 'unknown'),
                    },
                    last_check=datetime.utcnow().isoformat() + "Z",
                )
            else:
                return ComponentHealth(
                    name="llm",
                    status=ComponentStatus.DEGRADED,
                    latency_ms=latency,
                    message="LLM health check failed",
                    last_check=datetime.utcnow().isoformat() + "Z",
                )
                
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="llm",
                status=ComponentStatus.DOWN,
                latency_ms=latency,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    async def _check_tools(self) -> ComponentHealth:
        """Check tool registry health."""
        if not self._tool_registry:
            return ComponentHealth(
                name="tools",
                status=ComponentStatus.UNKNOWN,
                message="Tool registry not configured",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        
        try:
            tool_count = len(self._tool_registry.list_tools()) if hasattr(self._tool_registry, 'list_tools') else 0
            
            return ComponentHealth(
                name="tools",
                status=ComponentStatus.UP,
                details={
                    "registered_tools": tool_count,
                },
                last_check=datetime.utcnow().isoformat() + "Z",
            )
            
        except Exception as e:
            return ComponentHealth(
                name="tools",
                status=ComponentStatus.DOWN,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    async def _check_memory(self) -> ComponentHealth:
        """Check memory backend health."""
        if not self._memory_manager:
            return ComponentHealth(
                name="memory",
                status=ComponentStatus.UNKNOWN,
                message="Memory manager not configured",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        
        try:
            details = {}
            
            if hasattr(self._memory_manager, '_vector_store') and self._memory_manager._vector_store:
                details["vector_store"] = "connected"
            
            if hasattr(self._memory_manager, '_long_term_store') and self._memory_manager._long_term_store:
                details["long_term_store"] = "connected"
            
            return ComponentHealth(
                name="memory",
                status=ComponentStatus.UP,
                details=details,
                last_check=datetime.utcnow().isoformat() + "Z",
            )
            
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=ComponentStatus.DOWN,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    async def _check_mcp(self) -> ComponentHealth:
        """Check MCP connections health."""
        if not self._mcp_manager:
            return ComponentHealth(
                name="mcp",
                status=ComponentStatus.UNKNOWN,
                message="MCP manager not configured",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        
        try:
            connected_servers = len(self._mcp_manager._clients) if hasattr(self._mcp_manager, '_clients') else 0
            total_tools = len(self._mcp_manager.tools) if hasattr(self._mcp_manager, 'tools') else 0
            
            return ComponentHealth(
                name="mcp",
                status=ComponentStatus.UP if connected_servers > 0 else ComponentStatus.DEGRADED,
                details={
                    "connected_servers": connected_servers,
                    "available_tools": total_tools,
                },
                last_check=datetime.utcnow().isoformat() + "Z",
            )
            
        except Exception as e:
            return ComponentHealth(
                name="mcp",
                status=ComponentStatus.DOWN,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    async def _check_system_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            status = ComponentStatus.UP
            if percent_used > self.config.max_memory_percent:
                status = ComponentStatus.DOWN
            elif percent_used > self.config.max_memory_percent * 0.8:
                status = ComponentStatus.DEGRADED
            
            return ComponentHealth(
                name="system_memory",
                status=status,
                details={
                    "percent_used": round(percent_used, 2),
                    "available_mb": round(memory.available / (1024 * 1024), 2),
                    "total_mb": round(memory.total / (1024 * 1024), 2),
                },
                last_check=datetime.utcnow().isoformat() + "Z",
            )
            
        except ImportError:
            return ComponentHealth(
                name="system_memory",
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        except Exception as e:
            return ComponentHealth(
                name="system_memory",
                status=ComponentStatus.UNKNOWN,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    async def _check_disk(self) -> ComponentHealth:
        """Check disk space."""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            percent_used = disk.percent
            
            status = ComponentStatus.UP
            if percent_used > self.config.max_disk_percent:
                status = ComponentStatus.DOWN
            elif percent_used > self.config.max_disk_percent * 0.8:
                status = ComponentStatus.DEGRADED
            
            return ComponentHealth(
                name="disk",
                status=status,
                details={
                    "percent_used": round(percent_used, 2),
                    "free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                    "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                },
                last_check=datetime.utcnow().isoformat() + "Z",
            )
            
        except ImportError:
            return ComponentHealth(
                name="disk",
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed",
                last_check=datetime.utcnow().isoformat() + "Z",
            )
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=ComponentStatus.UNKNOWN,
                message=str(e),
                last_check=datetime.utcnow().isoformat() + "Z",
            )
    
    def create_router(self) -> APIRouter:
        """
        Create FastAPI router with health endpoints.
        
        Returns:
            FastAPI router with /health, /livez, /readyz, /startupz endpoints
        """
        router = APIRouter(tags=["Health"])
        
        @router.get("/health")
        async def health_endpoint():
            """
            Comprehensive health check endpoint.
            
            Returns detailed status of all components.
            """
            response = await self.check_health()
            
            status_code = 200
            if response.status == HealthStatus.UNHEALTHY:
                status_code = 503
            elif response.status == HealthStatus.DEGRADED:
                status_code = 200
            
            return JSONResponse(
                content=response.to_dict(include_details=self.config.include_details),
                status_code=status_code,
            )
        
        @router.get("/livez")
        async def liveness_endpoint():
            """
            Kubernetes liveness probe endpoint.
            
            Returns 200 if the process is alive, 503 otherwise.
            """
            if not self.config.liveness_enabled:
                return Response(status_code=404)
            
            is_live = await self.check_liveness()
            
            if is_live:
                return JSONResponse(
                    content={"status": "live"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"status": "dead"},
                    status_code=503,
                )
        
        @router.get("/readyz")
        async def readiness_endpoint():
            """
            Kubernetes readiness probe endpoint.
            
            Returns 200 if ready for traffic, 503 otherwise.
            """
            if not self.config.readiness_enabled:
                return Response(status_code=404)
            
            is_ready = await self.check_readiness()
            
            if is_ready:
                return JSONResponse(
                    content={"status": "ready"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"status": "not_ready"},
                    status_code=503,
                )
        
        @router.get("/startupz")
        async def startup_endpoint():
            """
            Kubernetes startup probe endpoint.
            
            Returns 200 if startup is complete, 503 otherwise.
            """
            if not self.config.startup_enabled:
                return Response(status_code=404)
            
            is_started = await self.check_startup()
            
            if is_started:
                return JSONResponse(
                    content={"status": "started"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"status": "starting"},
                    status_code=503,
                )
        
        @router.get("/ready")
        async def ready_endpoint():
            """Legacy readiness endpoint for backward compatibility."""
            is_ready = await self.check_readiness()
            if is_ready:
                return {"status": "ready"}
            return JSONResponse(
                content={"status": "not_ready"},
                status_code=503,
            )
        
        return router
