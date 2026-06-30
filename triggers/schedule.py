"""
Schedule Trigger
"""

import logging
import asyncio
from typing import Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Schedule trigger configuration."""
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    timezone: str = "UTC"
    enabled: bool = True


class ScheduleTrigger:
    """
    Schedule trigger for time-based agent invocation.
    
    Features:
    - Cron expressions
    - Fixed intervals
    - Timezone support
    """
    
    def __init__(self, config: ScheduleConfig, handler: Callable):
        self.config = config
        self.handler = handler
        self._task: Optional[asyncio.Task] = None
        self._active = False
    
    async def _run_interval(self):
        """Run handler at fixed intervals."""
        while self._active:
            try:
                await self.handler()
            except Exception as e:
                logger.error(f"Schedule handler error: {e}")
            await asyncio.sleep(self.config.interval_seconds or 60)
    
    async def _run_cron(self):
        """Run handler based on cron expression."""
        try:
            from croniter import croniter
        except ImportError:
            logger.error("croniter required for cron schedules: pip install croniter")
            return
        
        cron = croniter(self.config.cron, datetime.now())
        
        while self._active:
            next_run = cron.get_next(datetime)
            wait_seconds = (next_run - datetime.now()).total_seconds()
            
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            
            if self._active:
                try:
                    await self.handler()
                except Exception as e:
                    logger.error(f"Schedule handler error: {e}")
    
    def start(self):
        """Start the schedule trigger."""
        self._active = True
        
        if self.config.cron:
            self._task = asyncio.create_task(self._run_cron())
        elif self.config.interval_seconds:
            self._task = asyncio.create_task(self._run_interval())
        
        logger.info("Schedule trigger started")
    
    def stop(self):
        """Stop the schedule trigger."""
        self._active = False
        if self._task:
            self._task.cancel()
        logger.info("Schedule trigger stopped")
    
    @property
    def is_active(self) -> bool:
        return self._active


__all__ = ["ScheduleTrigger", "ScheduleConfig"]
