"""
LangGraph State Machine Definition for E-Commerce Refund Agent.
Integrates Chronos protection at node boundaries.
"""

from typing import Dict, Any, List, Optional
from chronos import ChronosHarness, ChronosAgentWrapper, ChronosLangGraphMiddleware
from .tools import tool_read_orders, tool_process_refund, tool_write_audit_script

class RefundAgentGraph:
    """
    Simulates a multi-node LangGraph execution loop with Chronos protection:
      Node 1: inspect_order
      Node 2: execute_refund
      Node 3: create_audit_trail
    """
    def __init__(self, harness: ChronosHarness):
        self.harness = harness
        self.wrapper = ChronosAgentWrapper(harness)
        self.middleware = ChronosLangGraphMiddleware(harness)

        # Wrap tools with Chronos transactional boundaries
        self.wrapped_read_orders = self.wrapper.wrap_tool(
            "read_orders",
            lambda args: tool_read_orders(args, harness.workspace_dir)
        )
        self.wrapped_process_refund = self.wrapper.wrap_tool(
            "process_refund",
            lambda args: tool_process_refund(args, harness.workspace_dir)
        )
        self.wrapped_write_audit = self.wrapper.wrap_tool(
            "write_audit_script",
            lambda args: tool_write_audit_script(args, harness.workspace_dir)
        )

    def run_node_inspect_order(self, order_id: str, prompt_stack: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Node 1: Inspect Order"""
        return self.wrapped_read_orders({"order_id": order_id}, prompt_stack)

    def run_node_execute_refund(self, order_id: str, amount: float, reason: str, prompt_stack: List[Dict[str, Any]], simulate_crash: bool = False) -> Dict[str, Any]:
        """Node 2: Execute Refund"""
        tool_args = {
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "simulate_crash": simulate_crash,
            "workspace_dir": self.harness.workspace_dir
        }
        return self.wrapped_process_refund(tool_args, prompt_stack)

    def run_node_create_audit(self, filename: str, code_content: str, prompt_stack: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Node 3: Create Audit Script"""
        tool_args = {
            "filename": filename,
            "code_content": code_content,
            "workspace_dir": self.harness.workspace_dir
        }
        return self.wrapped_write_audit(tool_args, prompt_stack)
