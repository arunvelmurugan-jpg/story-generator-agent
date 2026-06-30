"""
Code Executor Tool
"""

import logging
import sys
import io
import ast
from typing import Any, Dict, Optional
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutorConfig:
    timeout_seconds: int = 30
    max_output_length: int = 10000
    allow_imports: bool = False
    sandbox_mode: bool = True


@dataclass
class CodeResult:
    success: bool
    output: str
    return_value: Any
    error: Optional[str] = None


class CodeExecutorTool:
    """Code executor tool for running Python code safely."""
    
    name = "code_executor"
    description = "Execute Python code and return the result."
    
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float',
        'int', 'len', 'list', 'map', 'max', 'min', 'pow', 'print', 'range',
        'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip'
    }
    
    def __init__(self, config: Optional[CodeExecutorConfig] = None):
        self.config = config or CodeExecutorConfig()
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code"}},
                "required": ["code"]
            }
        }
    
    async def execute(self, code: str) -> CodeResult:
        try:
            if self.config.sandbox_mode:
                self._validate_code(code)
            
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            
            try:
                safe_builtins = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                               for name in self.SAFE_BUILTINS if (isinstance(__builtins__, dict) and name in __builtins__) or hasattr(__builtins__, name)}
                globals_dict = {'__builtins__': safe_builtins}
                locals_dict = {}
                exec(code, globals_dict, locals_dict)
                output = sys.stdout.getvalue()
                return CodeResult(success=True, output=output[:self.config.max_output_length],
                                return_value=locals_dict.get('result'))
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        except Exception as e:
            return CodeResult(success=False, output="", return_value=None, error=str(e))
    
    def _validate_code(self, code: str):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports not allowed")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'compile', '__import__', 'open'):
                    raise ValueError(f"'{node.func.id}' not allowed")


__all__ = ["CodeExecutorTool", "CodeExecutorConfig", "CodeResult"]
