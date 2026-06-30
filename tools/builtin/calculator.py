"""
Calculator Tool

Performs mathematical calculations safely.
Equivalent to n8n's Calculator tool.
"""

import math
import logging
import operator
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


SAFE_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '**': operator.pow,
    '^': operator.pow,
}

SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'log': math.log,
    'log10': math.log10,
    'log2': math.log2,
    'exp': math.exp,
    'floor': math.floor,
    'ceil': math.ceil,
    'factorial': math.factorial,
    'gcd': math.gcd,
    'pi': math.pi,
    'e': math.e,
}


@dataclass
class CalculatorResult:
    """Result of a calculation."""
    expression: str
    result: Union[int, float]
    success: bool
    error: Optional[str] = None


class CalculatorTool:
    """
    Calculator tool for mathematical operations.
    
    Features:
    - Basic arithmetic (+, -, *, /, //, %, **)
    - Mathematical functions (sqrt, sin, cos, log, etc.)
    - Safe evaluation (no code execution)
    - Expression parsing
    """
    
    name = "calculator"
    description = "Performs mathematical calculations. Input should be a mathematical expression."
    
    def __init__(self):
        self.safe_dict = {**SAFE_FUNCTIONS}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                    }
                },
                "required": ["expression"]
            }
        }
    
    async def execute(self, expression: str) -> CalculatorResult:
        """Execute a mathematical expression."""
        try:
            expression = expression.strip()
            
            expression = expression.replace('^', '**')
            
            result = self._safe_eval(expression)
            
            return CalculatorResult(
                expression=expression,
                result=result,
                success=True
            )
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return CalculatorResult(
                expression=expression,
                result=0,
                success=False,
                error=str(e)
            )
    
    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Safely evaluate a mathematical expression."""
        allowed_names = {**self.safe_dict}
        
        code = compile(expression, "<string>", "eval")
        
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"Use of '{name}' is not allowed")
        
        return eval(code, {"__builtins__": {}}, allowed_names)
    
    def __call__(self, expression: str) -> Union[int, float]:
        """Synchronous calculation."""
        result = self._safe_eval(expression)
        return result


def calculate(expression: str) -> Union[int, float]:
    """
    Convenience function for calculations.
    
    Args:
        expression: Mathematical expression
    
    Returns:
        Calculation result
    
    Examples:
        >>> calculate("2 + 2")
        4
        >>> calculate("sqrt(16)")
        4.0
        >>> calculate("sin(pi/2)")
        1.0
    """
    calc = CalculatorTool()
    return calc(expression)


__all__ = ["CalculatorTool", "CalculatorResult", "calculate"]
