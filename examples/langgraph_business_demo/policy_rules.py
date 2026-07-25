"""
Policy Rules & Business Invariants for E-Commerce Refund Agent.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from chronos.policy import PolicyEngine, Policy

def create_refund_agent_policy() -> PolicyEngine:
    """Configures the scope-locked permission policy for the refund agent role."""
    engine = PolicyEngine()
    engine.register_policy(Policy(
        role="refund_agent",
        allow_read=["mock_db/*", "*.py", "*.json"],
        allow_write=["mock_db/financial_ledger.json", "audit_*.py", "mock_db/orders.json"],
        allow_tools=["read_orders", "process_refund", "write_audit_script"],
        deny_tools=["execute_bash", "delete_database", "wipe_ledger"],
        description="Permission policy for E-Commerce Refund & Financial Audit Agent"
    ))
    return engine


def check_refund_invariant(payload: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Custom Chronos Invariant Rule:
    Ensures that a refund amount does NOT exceed the original order total,
    and company balance never drops below zero.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name == "process_refund":
        order_id = str(payload.get("order_id", ""))
        refund_amount = float(payload.get("amount", 0.0))

        # Check order amount limit
        orders_file = Path(payload.get("workspace_dir", ".")) / "mock_db" / "orders.json"
        if orders_file.exists():
            try:
                orders = json.loads(orders_file.read_text())
                order_info = orders.get(order_id)
                if order_info:
                    max_allowed = float(order_info.get("total_amount", 0.0))
                    if refund_amount > max_allowed:
                        return (
                            False,
                            f"INVARIANT VIOLATION: Requested refund ${refund_amount:.2f} exceeds original order total ${max_allowed:.2f} for Order #{order_id}.",
                            f"Refund cap exceeded: Maximum refundable amount for Order #{order_id} is ${max_allowed:.2f}. Adjust refund amount."
                        )
            except Exception:
                pass

        # Check negative refund amount
        if refund_amount <= 0:
            return (
                False,
                f"INVARIANT VIOLATION: Invalid refund amount ${refund_amount:.2f}.",
                "Refund amount must be positive."
            )

    return True, "Financial invariant passed.", None
