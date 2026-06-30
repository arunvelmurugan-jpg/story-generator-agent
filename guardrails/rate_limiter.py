"""
Rate Limiter for PHTN.AI Sub-Agent Framework

Implements rate limiting using multiple strategies:
- Token Bucket
- Sliding Window
- Fixed Window
- Leaky Bucket

Supports per-agent, per-tenant, and per-endpoint rate limiting.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    enabled: bool = True
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests_per_minute: int = 100
    burst_size: int = 20
    window_size_seconds: int = 60
    per_tenant: bool = True
    per_agent: bool = True
    per_endpoint: bool = False


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_after_seconds: float
    limit: int
    retry_after_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_after_seconds": self.reset_after_seconds,
            "limit": self.limit,
            "retry_after_seconds": self.retry_after_seconds,
        }
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(time.time() + self.reset_after_seconds)),
        }
        if self.retry_after_seconds:
            headers["Retry-After"] = str(int(self.retry_after_seconds))
        return headers


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(
        self,
        rate: float,
        capacity: int,
    ):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second
            capacity: Maximum bucket capacity (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """
        Try to acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            RateLimitResult
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.tokens),
                    reset_after_seconds=(self.capacity - self.tokens) / self.rate,
                    limit=self.capacity,
                )
            else:
                wait_time = (tokens - self.tokens) / self.rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=wait_time,
                    limit=self.capacity,
                    retry_after_seconds=wait_time,
                )


class SlidingWindow:
    """Sliding window rate limiter implementation."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        """
        Initialize sliding window.
        
        Args:
            limit: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: list = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = time.monotonic()
            window_start = now - self.window_seconds
            
            self.requests = [t for t in self.requests if t > window_start]
            
            if len(self.requests) < self.limit:
                self.requests.append(now)
                remaining = self.limit - len(self.requests)
                
                reset_after = self.window_seconds
                if self.requests:
                    oldest = min(self.requests)
                    reset_after = oldest + self.window_seconds - now
                
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_after_seconds=max(0, reset_after),
                    limit=self.limit,
                )
            else:
                oldest = min(self.requests)
                retry_after = oldest + self.window_seconds - now
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=retry_after,
                    limit=self.limit,
                    retry_after_seconds=retry_after,
                )


class FixedWindow:
    """Fixed window rate limiter implementation."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        """
        Initialize fixed window.
        
        Args:
            limit: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.count = 0
        self.window_start = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = time.monotonic()
            
            if now - self.window_start >= self.window_seconds:
                self.window_start = now
                self.count = 0
            
            if self.count < self.limit:
                self.count += 1
                remaining = self.limit - self.count
                reset_after = self.window_start + self.window_seconds - now
                
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_after_seconds=max(0, reset_after),
                    limit=self.limit,
                )
            else:
                reset_after = self.window_start + self.window_seconds - now
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=reset_after,
                    limit=self.limit,
                    retry_after_seconds=reset_after,
                )


class RateLimiter:
    """
    Multi-strategy rate limiter with per-key support.
    
    Supports rate limiting by:
    - Agent ID
    - Tenant ID
    - Endpoint
    - Custom keys
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self._limiters: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        logger.debug(f"RateLimiter initialized: {config.strategy.value}, {config.requests_per_minute}/min")
    
    def _get_limiter_key(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> str:
        """Generate limiter key based on configuration."""
        parts = []
        
        if self.config.per_tenant and tenant_id:
            parts.append(f"tenant:{tenant_id}")
        if self.config.per_agent and agent_id:
            parts.append(f"agent:{agent_id}")
        if self.config.per_endpoint and endpoint:
            parts.append(f"endpoint:{endpoint}")
        
        return ":".join(parts) if parts else "global"
    
    def _create_limiter(self):
        """Create a new limiter based on strategy."""
        strategy = self.config.strategy
        
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            rate = self.config.requests_per_minute / 60.0
            return TokenBucket(rate=rate, capacity=self.config.burst_size)
        
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindow(
                limit=self.config.requests_per_minute,
                window_seconds=self.config.window_size_seconds,
            )
        
        elif strategy == RateLimitStrategy.FIXED_WINDOW:
            return FixedWindow(
                limit=self.config.requests_per_minute,
                window_seconds=self.config.window_size_seconds,
            )
        
        else:
            rate = self.config.requests_per_minute / 60.0
            return TokenBucket(rate=rate, capacity=self.config.burst_size)
    
    async def check(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        tokens: int = 1,
    ) -> RateLimitResult:
        """
        Check if request is allowed.
        
        Args:
            agent_id: Optional agent ID
            tenant_id: Optional tenant ID
            endpoint: Optional endpoint
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitResult
        """
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                remaining=self.config.requests_per_minute,
                reset_after_seconds=0,
                limit=self.config.requests_per_minute,
            )
        
        key = self._get_limiter_key(agent_id, tenant_id, endpoint)
        
        async with self._lock:
            if key not in self._limiters:
                self._limiters[key] = self._create_limiter()
        
        limiter = self._limiters[key]
        
        if isinstance(limiter, TokenBucket):
            result = await limiter.acquire(tokens)
        else:
            result = await limiter.acquire()
        
        if not result.allowed:
            logger.warning(
                f"Rate limit exceeded for {key}: "
                f"retry after {result.retry_after_seconds:.2f}s"
            )
        
        return result
    
    async def reset(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """Reset rate limiter for a specific key."""
        key = self._get_limiter_key(agent_id, tenant_id, endpoint)
        
        async with self._lock:
            if key in self._limiters:
                del self._limiters[key]
    
    async def reset_all(self):
        """Reset all rate limiters."""
        async with self._lock:
            self._limiters.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "strategy": self.config.strategy.value,
            "requests_per_minute": self.config.requests_per_minute,
            "burst_size": self.config.burst_size,
            "active_limiters": len(self._limiters),
            "keys": list(self._limiters.keys()),
        }


_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create global rate limiter."""
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        if config is None:
            config = RateLimitConfig()
        _global_rate_limiter = RateLimiter(config)
    
    return _global_rate_limiter


def create_rate_limiter(config_dict: Optional[Dict[str, Any]] = None) -> RateLimiter:
    """Create rate limiter from config dict."""
    if config_dict is None:
        config_dict = {}
    
    strategy_str = config_dict.get("strategy", "token_bucket")
    try:
        strategy = RateLimitStrategy(strategy_str)
    except ValueError:
        strategy = RateLimitStrategy.TOKEN_BUCKET
    
    config = RateLimitConfig(
        enabled=config_dict.get("enabled", True),
        strategy=strategy,
        requests_per_minute=config_dict.get("requests_per_minute", 100),
        burst_size=config_dict.get("burst_size", 20),
        window_size_seconds=config_dict.get("window_size_seconds", 60),
        per_tenant=config_dict.get("per_tenant", True),
        per_agent=config_dict.get("per_agent", True),
        per_endpoint=config_dict.get("per_endpoint", False),
    )
    
    return RateLimiter(config)
