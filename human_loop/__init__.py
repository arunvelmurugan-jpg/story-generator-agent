"""
Human-in-the-Loop Module for PHTN.AI Sub-Agent Framework

Provides human approval workflows, input requests, and oversight mechanisms.
"""

from .manager import (
    HumanLoopManager,
    HumanLoopConfig,
    ApprovalRequest,
    ApprovalResponse,
    InputRequest,
    InputResponse,
    ApprovalStatus,
    ApprovalType,
)

__all__ = [
    "HumanLoopManager",
    "HumanLoopConfig",
    "ApprovalRequest",
    "ApprovalResponse",
    "InputRequest",
    "InputResponse",
    "ApprovalStatus",
    "ApprovalType",
]
