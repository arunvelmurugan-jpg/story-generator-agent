"""
Webhook Trigger
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class WebhookConfig:
    """Webhook trigger configuration."""
    path: str = "/webhook"
    methods: List[HttpMethod] = field(default_factory=lambda: [HttpMethod.POST])
    authentication: Optional[str] = None
    api_key: Optional[str] = None
    response_mode: str = "immediate"
    timeout_seconds: int = 30


class WebhookTrigger:
    """
    Webhook trigger for HTTP-based agent invocation.
    
    Features:
    - Multiple HTTP methods
    - Authentication support
    - Request validation
    - Response modes (immediate, deferred)
    """
    
    def __init__(self, config: WebhookConfig, handler: Callable):
        self.config = config
        self.handler = handler
        self._active = False
    
    async def handle_request(
        self,
        method: str,
        headers: Dict[str, str],
        body: Any,
        query_params: Dict[str, str]
    ) -> Dict[str, Any]:
        """Handle incoming webhook request."""
        if HttpMethod(method) not in self.config.methods:
            return {"error": "Method not allowed", "status": 405}
        
        if self.config.authentication == "api_key":
            api_key = headers.get("X-API-Key") or query_params.get("api_key")
            if api_key != self.config.api_key:
                return {"error": "Unauthorized", "status": 401}
        
        try:
            result = await self.handler(body)
            return {"result": result, "status": 200}
        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
            return {"error": str(e), "status": 500}
    
    def start(self):
        """Start the webhook trigger."""
        self._active = True
        logger.info(f"Webhook trigger started at {self.config.path}")
    
    def stop(self):
        """Stop the webhook trigger."""
        self._active = False
        logger.info("Webhook trigger stopped")
    
    @property
    def is_active(self) -> bool:
        return self._active


__all__ = ["WebhookTrigger", "WebhookConfig", "HttpMethod"]
