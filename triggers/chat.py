"""
Chat Trigger
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Chat trigger configuration."""
    allowed_users: Optional[List[str]] = None
    require_mention: bool = False
    prefix: Optional[str] = None
    max_history: int = 10


class ChatTrigger:
    """
    Chat trigger for conversational agent invocation.
    
    Features:
    - User filtering
    - Mention detection
    - Command prefixes
    - Conversation history
    """
    
    def __init__(self, config: ChatConfig, handler: Callable):
        self.config = config
        self.handler = handler
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._active = False
    
    async def handle_message(
        self,
        message: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Handle incoming chat message."""
        if self.config.allowed_users and user_id not in self.config.allowed_users:
            return None
        
        if self.config.prefix:
            if not message.startswith(self.config.prefix):
                return None
            message = message[len(self.config.prefix):].strip()
        
        if user_id not in self._history:
            self._history[user_id] = []
        
        self._history[user_id].append({"role": "user", "content": message})
        
        if len(self._history[user_id]) > self.config.max_history * 2:
            self._history[user_id] = self._history[user_id][-self.config.max_history * 2:]
        
        try:
            response = await self.handler(message, self._history[user_id])
            self._history[user_id].append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            logger.error(f"Chat handler error: {e}")
            return f"Error: {e}"
    
    def clear_history(self, user_id: Optional[str] = None):
        """Clear conversation history."""
        if user_id:
            self._history.pop(user_id, None)
        else:
            self._history.clear()
    
    def start(self):
        self._active = True
        logger.info("Chat trigger started")
    
    def stop(self):
        self._active = False
        logger.info("Chat trigger stopped")


__all__ = ["ChatTrigger", "ChatConfig"]
