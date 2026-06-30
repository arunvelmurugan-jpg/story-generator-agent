"""
Triggers Module - n8n Compatible

Provides agent trigger mechanisms:
- WebhookTrigger: HTTP webhook triggers
- ScheduleTrigger: Cron-based scheduling
- ChatTrigger: Chat message triggers
- MessageQueueTrigger: Queue-based triggers
- EventTrigger: Event-driven triggers

Aligned with n8n's trigger nodes.
"""

from .webhook import WebhookTrigger, WebhookConfig
from .schedule import ScheduleTrigger, ScheduleConfig
from .chat import ChatTrigger, ChatConfig
from .event import EventTrigger, EventConfig

__all__ = [
    "WebhookTrigger", "WebhookConfig",
    "ScheduleTrigger", "ScheduleConfig",
    "ChatTrigger", "ChatConfig",
    "EventTrigger", "EventConfig",
]
