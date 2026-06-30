"""
Auto-Fixing Output Parser

Automatically fixes malformed LLM outputs using an LLM.
Equivalent to n8n's Auto-fixing Output Parser node.
"""

import json
import logging
from typing import Any, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AutoFixingConfig:
    """Configuration for auto-fixing parser."""
    max_retries: int = 3
    retry_on_parse_error: bool = True
    retry_on_validation_error: bool = True
    include_original_error: bool = True
    fix_prompt_template: Optional[str] = None


class AutoFixingOutputParser:
    """
    Auto-fixing output parser that uses LLM to fix malformed outputs.
    
    Features:
    - Automatic retry with LLM-based fixing
    - Configurable retry limits
    - Custom fix prompt templates
    - Works with any base parser
    """
    
    DEFAULT_FIX_PROMPT = """The following output failed to parse correctly:

```
{output}
```

Error: {error}

Please fix the output to be valid. Return ONLY the corrected output, nothing else.

Expected format:
{format_instructions}"""
    
    def __init__(
        self,
        base_parser: Any,
        llm_call: Callable[[str], Awaitable[str]],
        config: Optional[AutoFixingConfig] = None
    ):
        """
        Initialize auto-fixing parser.
        
        Args:
            base_parser: The underlying parser to use
            llm_call: Async function to call LLM for fixing
            config: Parser configuration
        """
        self.base_parser = base_parser
        self.llm_call = llm_call
        self.config = config or AutoFixingConfig()
        self.fix_prompt = self.config.fix_prompt_template or self.DEFAULT_FIX_PROMPT
    
    def get_format_instructions(self) -> str:
        """Get format instructions from base parser."""
        if hasattr(self.base_parser, "get_format_instructions"):
            return self.base_parser.get_format_instructions()
        return ""
    
    async def parse(self, text: str) -> Any:
        """Parse with auto-fixing on failure."""
        last_error = None
        current_text = text
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.base_parser.parse(current_text)
            except Exception as e:
                last_error = e
                logger.warning(f"Parse attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    should_retry = (
                        (self.config.retry_on_parse_error and "parse" in str(e).lower()) or
                        (self.config.retry_on_validation_error and "valid" in str(e).lower()) or
                        True
                    )
                    
                    if should_retry:
                        current_text = await self._fix_output(current_text, str(e))
        
        raise ValueError(f"Failed to parse after {self.config.max_retries} retries: {last_error}")
    
    async def _fix_output(self, output: str, error: str) -> str:
        """Use LLM to fix malformed output."""
        fix_prompt = self.fix_prompt.format(
            output=output,
            error=error if self.config.include_original_error else "Parse error",
            format_instructions=self.get_format_instructions()
        )
        
        try:
            fixed = await self.llm_call(fix_prompt)
            logger.info("LLM provided fixed output")
            return fixed
        except Exception as e:
            logger.error(f"LLM fix failed: {e}")
            return output
    
    def parse_sync(self, text: str) -> Any:
        """Synchronous parse without auto-fixing."""
        return self.base_parser.parse(text)


class RetryWithErrorOutputParser:
    """
    Parser that retries with error context.
    
    Similar to AutoFixingOutputParser but includes the original
    prompt and error in retry attempts.
    """
    
    def __init__(
        self,
        base_parser: Any,
        llm_call: Callable[[str], Awaitable[str]],
        max_retries: int = 3
    ):
        self.base_parser = base_parser
        self.llm_call = llm_call
        self.max_retries = max_retries
    
    async def parse_with_prompt(self, text: str, original_prompt: str) -> Any:
        """Parse with retry using original prompt context."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self.base_parser.parse(text)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    retry_prompt = f"""{original_prompt}

Previous attempt failed with error: {e}

Please try again and ensure the output is valid."""
                    
                    text = await self.llm_call(retry_prompt)
        
        raise ValueError(f"Failed after {self.max_retries} retries: {last_error}")


def create_auto_fixing_parser(
    base_parser: Any,
    llm_call: Callable[[str], Awaitable[str]],
    max_retries: int = 3,
    **kwargs
) -> AutoFixingOutputParser:
    """Factory function to create auto-fixing parser."""
    config = AutoFixingConfig(max_retries=max_retries, **kwargs)
    return AutoFixingOutputParser(base_parser, llm_call, config)


__all__ = [
    "AutoFixingOutputParser",
    "AutoFixingConfig",
    "RetryWithErrorOutputParser",
    "create_auto_fixing_parser"
]
