"""
Access Control for PHTN.AI Sub-Agent Framework

Implements RBAC and ABAC for agent security.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Permission:
    """Permission definition."""
    resource: str
    action: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, resource: str, action: str) -> bool:
        """Check if permission matches request."""
        resource_match = self.resource == "*" or self.resource == resource
        action_match = self.action == "*" or self.action == action
        return resource_match and action_match


@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    permissions: List[Permission] = field(default_factory=list)
    description: str = ""
    
    def has_permission(self, resource: str, action: str) -> bool:
        """Check if role has permission."""
        return any(p.matches(resource, action) for p in self.permissions)


@dataclass
class AccessRequest:
    """Access request for authorization."""
    subject: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessDecision:
    """Access decision result."""
    allowed: bool
    reason: str = ""
    matched_policy: Optional[str] = None


class AccessController:
    """
    Access control with RBAC and ABAC support.
    
    Features:
    - Role-based access control
    - Attribute-based policies
    - Permission inheritance
    - Audit logging integration
    """
    
    BUILTIN_ROLES = {
        "admin": Role(
            name="admin",
            description="Full access to all resources",
            permissions=[Permission(resource="*", action="*")],
        ),
        "operator": Role(
            name="operator",
            description="Execute and monitor agents",
            permissions=[
                Permission(resource="agent", action="execute"),
                Permission(resource="agent", action="read"),
                Permission(resource="tool", action="execute"),
            ],
        ),
        "viewer": Role(
            name="viewer",
            description="Read-only access",
            permissions=[
                Permission(resource="*", action="read"),
            ],
        ),
    }
    
    def __init__(
        self,
        rbac_config: Optional[Dict[str, Any]] = None,
        abac_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AccessController.
        
        Args:
            rbac_config: RBAC configuration
            abac_config: ABAC configuration
        """
        self.rbac_config = rbac_config or {}
        self.abac_config = abac_config or {}
        
        self._roles: Dict[str, Role] = dict(self.BUILTIN_ROLES)
        self._subject_roles: Dict[str, Set[str]] = {}
        self._abac_policies: List[Dict[str, Any]] = []
        
        self._load_config()
        
        logger.debug("AccessController initialized")
    
    def _load_config(self):
        """Load configuration."""
        if self.rbac_config.get("enabled", False):
            for role_def in self.rbac_config.get("roles", []):
                role = Role(
                    name=role_def.get("name"),
                    description=role_def.get("description", ""),
                    permissions=[
                        Permission(
                            resource=p.get("resource", "*"),
                            action=p.get("action", "*"),
                            conditions=p.get("conditions", {}),
                        )
                        for p in role_def.get("permissions", [])
                    ],
                )
                self._roles[role.name] = role
        
        if self.abac_config.get("enabled", False):
            self._abac_policies = self.abac_config.get("policies", [])
    
    def assign_role(self, subject: str, role_name: str) -> bool:
        """Assign role to subject."""
        if role_name not in self._roles:
            logger.warning(f"Role not found: {role_name}")
            return False
        
        if subject not in self._subject_roles:
            self._subject_roles[subject] = set()
        
        self._subject_roles[subject].add(role_name)
        logger.info(f"Role '{role_name}' assigned to '{subject}'")
        return True
    
    def revoke_role(self, subject: str, role_name: str) -> bool:
        """Revoke role from subject."""
        if subject in self._subject_roles:
            self._subject_roles[subject].discard(role_name)
            return True
        return False
    
    def authorize(self, request: AccessRequest) -> AccessDecision:
        """
        Authorize access request.
        
        Args:
            request: Access request
            
        Returns:
            AccessDecision
        """
        if self.rbac_config.get("enabled", False):
            rbac_decision = self._check_rbac(request)
            if rbac_decision.allowed:
                return rbac_decision
        
        if self.abac_config.get("enabled", False):
            abac_decision = self._check_abac(request)
            if abac_decision.allowed:
                return abac_decision
        
        if not self.rbac_config.get("enabled") and not self.abac_config.get("enabled"):
            return AccessDecision(allowed=True, reason="Access control disabled")
        
        return AccessDecision(
            allowed=False,
            reason="No matching policy found",
        )
    
    def _check_rbac(self, request: AccessRequest) -> AccessDecision:
        """Check RBAC policies."""
        subject_roles = self._subject_roles.get(request.subject, set())
        
        for role_name in subject_roles:
            role = self._roles.get(role_name)
            if role and role.has_permission(request.resource, request.action):
                return AccessDecision(
                    allowed=True,
                    reason=f"Allowed by role: {role_name}",
                    matched_policy=f"rbac:{role_name}",
                )
        
        return AccessDecision(
            allowed=False,
            reason="No matching RBAC role",
        )
    
    def _check_abac(self, request: AccessRequest) -> AccessDecision:
        """Check ABAC policies."""
        for policy in self._abac_policies:
            if self._evaluate_abac_policy(policy, request):
                return AccessDecision(
                    allowed=True,
                    reason=f"Allowed by ABAC policy: {policy.get('name', 'unnamed')}",
                    matched_policy=f"abac:{policy.get('name', 'unnamed')}",
                )
        
        return AccessDecision(
            allowed=False,
            reason="No matching ABAC policy",
        )
    
    def _evaluate_abac_policy(
        self,
        policy: Dict[str, Any],
        request: AccessRequest,
    ) -> bool:
        """Evaluate ABAC policy against request."""
        conditions = policy.get("conditions", {})
        
        if "subject" in conditions:
            if request.subject not in conditions["subject"]:
                return False
        
        if "resource" in conditions:
            if request.resource not in conditions["resource"]:
                return False
        
        if "action" in conditions:
            if request.action not in conditions["action"]:
                return False
        
        if "context" in conditions:
            for key, expected in conditions["context"].items():
                actual = request.context.get(key)
                if actual != expected:
                    return False
        
        return True
    
    def get_subject_permissions(self, subject: str) -> List[Permission]:
        """Get all permissions for a subject."""
        permissions = []
        for role_name in self._subject_roles.get(subject, set()):
            role = self._roles.get(role_name)
            if role:
                permissions.extend(role.permissions)
        return permissions
