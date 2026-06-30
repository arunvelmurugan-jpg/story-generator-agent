"""Config-driven execution engine for the Story Generator Agent.

Reads PHTN-AGENT.json and dispatches to StoryGeneratorEngine.
Category "STORY_GENERATOR" maps to the engine that contains
the business logic ported from the original main.py /run endpoint.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .engines.story_generator import StoryGeneratorEngine

logger = logging.getLogger(__name__)

_ENGINE_MAP = {
    "STORY_GENERATOR": StoryGeneratorEngine,
}


class ConfigEngine:
    """Reads PHTN-AGENT.json once and creates the appropriate execution engine."""

    def __init__(self, config_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        if config:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.agent_id = self.config.get("agent_id", "unknown")
        self.name = self.config.get("name", "Unknown Agent")
        self.category = self.config.get("category", "").upper()

        engine_cls = _ENGINE_MAP.get(self.category)
        if not engine_cls:
            raise ValueError(
                f"Unsupported agent category '{self.category}'. "
                f"Supported: {list(_ENGINE_MAP.keys())}"
            )

        self.engine = engine_cls(self.config)
        logger.info(f"ConfigEngine initialized: {self.name} ({self.agent_id}) category={self.category}")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute the agent's logic based entirely on PHTN-AGENT.json config."""
        return await self.engine.execute(input_data, context)
