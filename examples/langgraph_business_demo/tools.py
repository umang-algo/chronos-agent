"""
Business Tools for E-Commerce Refund Agent.
All tools are wrapped via Chronos to enforce atomic transaction boundaries.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

def tool_read_orders(args: Dict[str, Any], workspace_dir: str) -> Dict[str, Any]:
    """Reads order details from mock_db/orders.json."""
    order_id = str(args.get("order_id", ""))
    orders_path = Path(workspace_dir) / "mock_db" / "orders.json"

    if not orders_path.exists():
        return {"success": False, "error": "Orders database missing."}

    orders = json.loads(orders_path.read_text())
    if order_id not in orders:
        return {"success": False, "error": f"Order #{order_id} not found."}

    return {"success": True, "order": orders[order_id]}


def tool_process_refund(args: Dict[str, Any], workspace_dir: str) -> Dict[str, Any]:
    """
    Processes a refund by updating financial_ledger.json.
    Can simulate a database crash if args contain 'simulate_crash'=True.
    """
    order_id = str(args.get("order_id", ""))
    amount = float(args.get("amount", 0.0))
    reason = args.get("reason", "Customer requested refund")
    simulate_crash = args.get("simulate_crash", False)

    ledger_path = Path(workspace_dir) / "mock_db" / "financial_ledger.json"
    if not ledger_path.exists():
        return {"success": False, "error": "Ledger missing."}

    # Simulate database crash mid-write if requested
    if simulate_crash:
        # Corrupt file on disk first to simulate partial write failure
        ledger_path.write_text("CORRUPTED_PARTIAL_WRITE_FILE_DATA")
        raise RuntimeError("Database Connection Timeout: Lost connection to ledger DB during transaction commit!")

    ledger = json.loads(ledger_path.read_text())
    ledger["company_balance"] -= amount
    ledger["total_refunds_processed"] += amount
    ledger["refund_history"].append({
        "order_id": order_id,
        "amount": amount,
        "reason": reason
    })

    ledger_path.write_text(json.dumps(ledger, indent=2))
    return {
        "success": True,
        "message": f"Processed ${amount:.2f} refund for Order #{order_id}",
        "new_balance": ledger["company_balance"]
    }


def tool_write_audit_script(args: Dict[str, Any], workspace_dir: str) -> Dict[str, Any]:
    """Generates an audit script (e.g. audit_refund_1001.py)."""
    filename = args.get("filename", "audit_refund.py")
    code_content = args.get("code_content", "")

    file_path = Path(workspace_dir) / filename
    file_path.write_text(code_content)

    return {"success": True, "filepath": str(file_path)}
