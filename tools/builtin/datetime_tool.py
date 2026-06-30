"""
DateTime Tool

Date and time operations.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DateTimeTool:
    """
    DateTime tool for date/time operations.
    
    Features:
    - Current date/time
    - Date parsing
    - Date formatting
    - Date arithmetic
    - Timezone conversion
    """
    
    name = "datetime"
    description = "Get current date/time, parse dates, or perform date calculations."
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "parse", "format", "add", "diff", "convert_tz"],
                        "description": "Operation to perform"
                    },
                    "date_string": {"type": "string", "description": "Date string to parse"},
                    "format": {"type": "string", "description": "Date format string"},
                    "days": {"type": "integer", "description": "Days to add/subtract"},
                    "hours": {"type": "integer", "description": "Hours to add/subtract"},
                    "timezone": {"type": "string", "description": "Target timezone"}
                },
                "required": ["operation"]
            }
        }
    
    async def execute(
        self,
        operation: str,
        date_string: Optional[str] = None,
        format: Optional[str] = None,
        days: int = 0,
        hours: int = 0,
        tz: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute datetime operation."""
        try:
            if operation == "now":
                now = datetime.now(timezone.utc)
                return {
                    "iso": now.isoformat(),
                    "timestamp": now.timestamp(),
                    "formatted": now.strftime(format or "%Y-%m-%d %H:%M:%S UTC")
                }
            
            elif operation == "parse":
                if not date_string:
                    return {"error": "date_string required"}
                
                if format:
                    dt = datetime.strptime(date_string, format)
                else:
                    from dateutil import parser
                    dt = parser.parse(date_string)
                
                return {
                    "iso": dt.isoformat(),
                    "timestamp": dt.timestamp() if dt.tzinfo else None,
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "hour": dt.hour,
                    "minute": dt.minute,
                    "second": dt.second
                }
            
            elif operation == "format":
                if not date_string:
                    dt = datetime.now(timezone.utc)
                else:
                    from dateutil import parser
                    dt = parser.parse(date_string)
                
                return {
                    "formatted": dt.strftime(format or "%Y-%m-%d %H:%M:%S")
                }
            
            elif operation == "add":
                if date_string:
                    from dateutil import parser
                    dt = parser.parse(date_string)
                else:
                    dt = datetime.now(timezone.utc)
                
                result = dt + timedelta(days=days, hours=hours)
                return {
                    "original": dt.isoformat(),
                    "result": result.isoformat(),
                    "added_days": days,
                    "added_hours": hours
                }
            
            elif operation == "diff":
                if not date_string:
                    return {"error": "date_string required for diff"}
                
                from dateutil import parser
                dt = parser.parse(date_string)
                now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
                diff = now - dt
                
                return {
                    "days": diff.days,
                    "seconds": diff.seconds,
                    "total_seconds": diff.total_seconds(),
                    "total_hours": diff.total_seconds() / 3600,
                    "total_days": diff.total_seconds() / 86400
                }
            
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except ImportError:
            return {"error": "python-dateutil required: pip install python-dateutil"}
        except Exception as e:
            return {"error": str(e)}


__all__ = ["DateTimeTool"]
