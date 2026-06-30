"""
Event Trigger
"""

import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EventSource(str, Enum):
    INTERNAL = "internal"
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
    SQS = "sqs"
    PUBSUB = "pubsub"
    REDIS = "redis"


@dataclass
class EventConfig:
    """Event trigger configuration."""
    source: EventSource = EventSource.INTERNAL
    topics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    connection_string: Optional[str] = None


class EventTrigger:
    """
    Event trigger for event-driven agent invocation.
    
    Features:
    - Multiple event sources
    - Topic filtering
    - Event pattern matching
    """
    
    def __init__(self, config: EventConfig, handler: Callable):
        self.config = config
        self.handler = handler
        self._active = False
        self._queue: asyncio.Queue = asyncio.Queue()
    
    async def emit(self, event_type: str, data: Any):
        """Emit an event (for internal source)."""
        if self.config.source == EventSource.INTERNAL:
            if not self.config.topics or event_type in self.config.topics:
                await self._queue.put({"type": event_type, "data": data})
    
    async def _process_events(self):
        """Process events from queue."""
        while self._active:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                if self._matches_filters(event):
                    try:
                        await self.handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
            except asyncio.TimeoutError:
                continue
    
    def _matches_filters(self, event: Dict[str, Any]) -> bool:
        """Check if event matches configured filters."""
        if not self.config.filters:
            return True
        
        for key, value in self.config.filters.items():
            if event.get(key) != value:
                return False
        return True
    
    def start(self):
        """Start the event trigger."""
        self._active = True
        asyncio.create_task(self._process_events())
        logger.info(f"Event trigger started for source: {self.config.source}")
    
    def stop(self):
        """Stop the event trigger."""
        self._active = False
        logger.info("Event trigger stopped")
    
    @property
    def is_active(self) -> bool:
        return self._active


__all__ = ["EventTrigger", "EventConfig", "EventSource"]
