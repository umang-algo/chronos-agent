"""
Scope-Locked Invariant Policy Engine for Chronos-Agent.

Declarative, role-based permission lattices that enforce which agents
can read, write, or execute which paths and tools — at the code level,
not the DevOps/IAM level.

Usage:
    from chronos.policy import PolicyEngine, Policy, Permission

    engine = PolicyEngine()

    engine.register_policy(Policy(
        role="finance_agent",
        allow_read=["ledger/*", "reports/*"],
        allow_write=["reports/*"],
        allow_tools=["read_file", "write_report"],
        deny_tools=["execute_bash", "delete_file"],
    ))

    violations = engine.enforce(
        role="finance_agent",
        tool_name="execute_bash",
        tool_args={"command": "rm -rf /var/db"}
    )
    # -> [PolicyViolation: role 'finance_agent' is denied tool 'execute_bash']
"""

import fnmatch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """
    A declarative permission policy for an agent role.

    - allow_read:  Glob patterns for paths the role may read.
    - allow_write: Glob patterns for paths the role may write/modify/delete.
    - allow_tools: Explicit tool names the role may call.
    - deny_tools:  Explicit tool names always blocked regardless of allow_tools.
    - allow_all_tools: If True, all tools allowed except deny_tools.
    """
    role: str
    allow_read:       List[str] = field(default_factory=list)
    allow_write:      List[str] = field(default_factory=list)
    allow_tools:      List[str] = field(default_factory=list)
    deny_tools:       List[str] = field(default_factory=list)
    allow_all_tools:  bool = False
    description:      str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "allow_read": self.allow_read,
            "allow_write": self.allow_write,
            "allow_tools": self.allow_tools,
            "deny_tools": self.deny_tools,
            "allow_all_tools": self.allow_all_tools,
            "description": self.description,
        }


@dataclass
class PolicyViolation:
    """Describes a single policy enforcement failure."""
    role:       str
    tool_name:  str
    violation:  str   # Short machine-readable code
    message:    str
    hint:       str = ""

    def __str__(self):
        return f"[PolicyViolation] role='{self.role}' tool='{self.tool_name}': {self.message}"


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Evaluates agent actions against declarative scope-locked policies.

    Integrates directly with InvariantVerifier: register it as a pre-flight
    rule so every tool call is policy-checked before execution.
    """

    def __init__(self):
        self._policies: Dict[str, Policy] = {}

    def register_policy(self, policy: Policy) -> None:
        """Register a policy for an agent role (overwrites any existing policy)."""
        self._policies[policy.role] = policy

    def get_policy(self, role: str) -> Optional[Policy]:
        return self._policies.get(role)

    def enforce(
        self,
        role: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> List[PolicyViolation]:
        """
        Evaluates all policy rules for the given role + tool call.
        Returns a list of PolicyViolations (empty = all clear).
        """
        violations: List[PolicyViolation] = []
        policy = self._policies.get(role)

        if policy is None:
            # No policy registered = default deny everything
            violations.append(PolicyViolation(
                role=role,
                tool_name=tool_name,
                violation="NO_POLICY",
                message=f"No policy registered for role '{role}'. All actions blocked by default.",
                hint=f"Register a Policy for role '{role}' via PolicyEngine.register_policy()."
            ))
            return violations

        # 1. Check tool-level deny list first (always wins)
        if tool_name in policy.deny_tools:
            violations.append(PolicyViolation(
                role=role,
                tool_name=tool_name,
                violation="TOOL_DENIED",
                message=f"Tool '{tool_name}' is explicitly denied for role '{role}'.",
                hint=f"Role '{role}' must not use '{tool_name}'. Use an allowed tool instead."
            ))

        # 2. Check tool-level allow list
        elif not policy.allow_all_tools and tool_name not in policy.allow_tools:
            violations.append(PolicyViolation(
                role=role,
                tool_name=tool_name,
                violation="TOOL_NOT_ALLOWED",
                message=f"Tool '{tool_name}' is not in the allow_tools list for role '{role}'.",
                hint=f"Allowed tools for '{role}': {policy.allow_tools}. Request a policy update if needed."
            ))

        # 3. Check path-level write scope (only when tool passed tool-level checks)
        write_path = tool_args.get("filepath") or tool_args.get("path") or tool_args.get("file")
        if write_path and tool_name in ("write_file", "delete_file", "create_file", "move_file") and not violations:
            if policy.allow_write and not self._matches_any(str(write_path), policy.allow_write):
                violations.append(PolicyViolation(
                    role=role,
                    tool_name=tool_name,
                    violation="WRITE_SCOPE_VIOLATION",
                    message=f"Role '{role}' attempted to write to '{write_path}' which is outside its write scope.",
                    hint=f"Write scope for '{role}': {policy.allow_write}. Choose a path within scope."
                ))

        # 4. Check path-level read scope (only when tool passed tool-level checks)
        read_path = tool_args.get("filepath") or tool_args.get("path") or tool_args.get("file")
        if read_path and tool_name in ("read_file", "list_dir") and not violations:
            if policy.allow_read and not self._matches_any(str(read_path), policy.allow_read):
                violations.append(PolicyViolation(
                    role=role,
                    tool_name=tool_name,
                    violation="READ_SCOPE_VIOLATION",
                    message=f"Role '{role}' attempted to read '{read_path}' which is outside its read scope.",
                    hint=f"Read scope for '{role}': {policy.allow_read}. Choose a path within scope."
                ))

        return violations

    def _matches_any(self, path: str, patterns: List[str]) -> bool:
        """Returns True if path matches any of the given glob patterns (hardened against path traversal)."""
        import os
        normalized_path = os.path.normpath(str(path)).lstrip("/\\")
        for pattern in patterns:
            norm_pattern = os.path.normpath(str(pattern)).lstrip("/\\")
            if fnmatch.fnmatch(normalized_path, norm_pattern) or normalized_path.startswith(norm_pattern.rstrip("*")):
                return True
        return False

    def build_verifier_rule(self, role: str):
        """
        Returns a pre-flight rule function compatible with InvariantVerifier.register_pre_flight_rule().
        Plug this directly into your Chronos harness verifier.

        Example:
            engine = PolicyEngine()
            engine.register_policy(Policy(role="my_agent", ...))
            harness.verifier.register_pre_flight_rule(
                "policy_engine",
                engine.build_verifier_rule("my_agent")
            )
        """
        def policy_pre_flight_rule(payload: Dict[str, Any]):
            tool_name = payload.get("tool_name", "unknown")
            violations = self.enforce(role=role, tool_name=tool_name, tool_args=payload)
            if violations:
                v = violations[0]
                return (False, v.message, v.hint)
            return (True, f"Policy check passed for role '{role}'.", None)

        return policy_pre_flight_rule
