"""
Pydantic Output Parser
"""

import json
import re
import logging
from typing import Any, Dict, Generic, List, Type, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class PydanticOutputParser(Generic[T]):
    """Parses LLM outputs into Pydantic models."""
    
    def __init__(self, pydantic_class: Type[T]):
        self.pydantic_class = pydantic_class
    
    def get_format_instructions(self) -> str:
        try:
            schema = self.pydantic_class.model_json_schema()
        except AttributeError:
            schema = self.pydantic_class.schema()
        return f"Return JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
    
    def parse(self, text: str) -> T:
        json_str = self._extract_json(text)
        data = json.loads(json_str)
        try:
            return self.pydantic_class.model_validate(data)
        except AttributeError:
            return self.pydantic_class.parse_obj(data)
    
    def _extract_json(self, text: str) -> str:
        patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```', r'\{[\s\S]*\}']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1) if '```' in pattern else match.group()
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue
        return text.strip()
    
    def parse_list(self, text: str) -> List[T]:
        json_str = self._extract_json(text)
        data = json.loads(json_str)
        if not isinstance(data, list):
            data = [data]
        results = []
        for item in data:
            try:
                try:
                    results.append(self.pydantic_class.model_validate(item))
                except AttributeError:
                    results.append(self.pydantic_class.parse_obj(item))
            except Exception as e:
                logger.warning(f"Failed to parse item: {e}")
        return results
    
    @classmethod
    def from_model(cls, model_class: Type[T]) -> "PydanticOutputParser[T]":
        return cls(model_class)


def create_pydantic_parser(model_class: Type[T]) -> PydanticOutputParser[T]:
    return PydanticOutputParser(model_class)


__all__ = ["PydanticOutputParser", "create_pydantic_parser"]
