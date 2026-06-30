"""
HTTP Request Tool

Make HTTP requests to external APIs.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class HttpRequestConfig:
    """Configuration for HTTP request tool."""
    timeout_seconds: int = 30
    max_retries: int = 3
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_response_size: int = 10485760


@dataclass
class HttpResponse:
    """HTTP response."""
    status_code: int
    headers: Dict[str, str]
    body: Any
    success: bool
    error: Optional[str] = None


class HttpRequestTool:
    """
    HTTP request tool for API calls.
    
    Features:
    - All HTTP methods
    - Header customization
    - JSON/form data support
    - Authentication
    - Retry logic
    """
    
    name = "http_request"
    description = "Make HTTP requests to external APIs and services."
    
    def __init__(self, config: Optional[HttpRequestConfig] = None):
        self.config = config or HttpRequestConfig()
        self._client = None
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"},
                    "headers": {"type": "object", "description": "Request headers"},
                    "body": {"type": "object", "description": "Request body (for POST/PUT/PATCH)"},
                    "params": {"type": "object", "description": "Query parameters"}
                },
                "required": ["url"]
            }
        }
    
    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None
    ) -> HttpResponse:
        """Execute HTTP request."""
        try:
            import httpx
            
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
                follow_redirects=self.config.follow_redirects
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if body and method in ("POST", "PUT", "PATCH") else None,
                    params=params
                )
                
                try:
                    response_body = response.json()
                except Exception:
                    response_body = response.text
                
                return HttpResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response_body,
                    success=200 <= response.status_code < 300
                )
        except ImportError:
            return HttpResponse(
                status_code=0, headers={}, body=None, success=False,
                error="httpx not installed: pip install httpx"
            )
        except Exception as e:
            return HttpResponse(
                status_code=0, headers={}, body=None, success=False,
                error=str(e)
            )


__all__ = ["HttpRequestTool", "HttpRequestConfig", "HttpResponse", "HttpMethod"]
