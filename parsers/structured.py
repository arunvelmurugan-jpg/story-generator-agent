"""
Structured Output Parser

Parses LLM outputs into structured JSON based on a schema.
Equivalent to n8n's Structured Output Parser node.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output parser."""
    schema: Dict[str, Any] = field(default_factory=dict)
    strict: bool = True
    extract_json: bool = True
    allow_partial: bool = False
    default_values: Dict[str, Any] = field(default_factory=dict)


class StructuredOutputParser:
    """
    Parses LLM outputs into structured JSON.
    
    Features:
    - JSON schema validation
    - Automatic JSON extraction from text
    - Partial parsing support
    - Default value injection
    - Type coercion
    """
    
    def __init__(self, config: Optional[StructuredOutputConfig] = None):
        self.config = config or StructuredOutputConfig()
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM."""
        schema_str = json.dumps(self.config.schema, indent=2)
        return f"""You must respond with a valid JSON object that matches this schema:

```json
{schema_str}
```

Important:
- Return ONLY the JSON object, no additional text
- Ensure all required fields are present
- Use the exact field names specified
- Match the data types exactly"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured data."""
        json_str = self._extract_json(text) if self.config.extract_json else text
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.config.allow_partial:
                parsed = self._partial_parse(json_str)
            else:
                raise ValueError(f"Failed to parse JSON: {e}")
        
        if self.config.strict:
            self._validate_schema(parsed)
        
        result = {**self.config.default_values, **parsed}
        
        return result
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
            r'\[[\s\S]*\]'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        return text.strip()
    
    def _partial_parse(self, text: str) -> Dict[str, Any]:
        """Attempt to parse partial/malformed JSON."""
        result = {}
        
        key_value_pattern = r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null|\[[\s\S]*?\]|\{[\s\S]*?\})'
        matches = re.findall(key_value_pattern, text)
        
        for key, value in matches:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value.strip('"')
        
        return result
    
    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """Validate data against schema."""
        if not self.config.schema:
            return
        
        properties = self.config.schema.get("properties", {})
        required = self.config.schema.get("required", [])
        
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")
        
        for field_name, field_schema in properties.items():
            if field_name in data:
                self._validate_field(field_name, data[field_name], field_schema)
    
    def _validate_field(self, name: str, value: Any, schema: Dict[str, Any]) -> None:
        """Validate a single field against its schema."""
        expected_type = schema.get("type")
        
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        if expected_type and expected_type in type_mapping:
            expected = type_mapping[expected_type]
            if not isinstance(value, expected):
                raise ValueError(
                    f"Field '{name}' expected type {expected_type}, got {type(value).__name__}"
                )
        
        if expected_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise ValueError(f"Field '{name}' is too short")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                raise ValueError(f"Field '{name}' is too long")
            if "enum" in schema and value not in schema["enum"]:
                raise ValueError(f"Field '{name}' must be one of {schema['enum']}")
        
        if expected_type in ("number", "integer"):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"Field '{name}' is below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"Field '{name}' is above maximum")
        
        if expected_type == "array":
            if "minItems" in schema and len(value) < schema["minItems"]:
                raise ValueError(f"Field '{name}' has too few items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                raise ValueError(f"Field '{name}' has too many items")
    
    @classmethod
    def from_schema(cls, schema: Dict[str, Any], **kwargs) -> "StructuredOutputParser":
        """Create parser from JSON schema."""
        config = StructuredOutputConfig(schema=schema, **kwargs)
        return cls(config)
    
    @classmethod
    def from_response_schemas(
        cls,
        response_schemas: List[Dict[str, Any]]
    ) -> "StructuredOutputParser":
        """Create parser from response schema definitions."""
        properties = {}
        required = []
        
        for schema in response_schemas:
            name = schema["name"]
            properties[name] = {
                "type": schema.get("type", "string"),
                "description": schema.get("description", "")
            }
            if schema.get("required", True):
                required.append(name)
        
        full_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        return cls.from_schema(full_schema)


def create_structured_parser(
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    **kwargs
) -> StructuredOutputParser:
    """Factory function to create structured output parser."""
    config = StructuredOutputConfig(
        schema=schema or {},
        strict=strict,
        **kwargs
    )
    return StructuredOutputParser(config)


__all__ = ["StructuredOutputParser", "StructuredOutputConfig", "create_structured_parser"]
