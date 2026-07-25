"""
OpenAI Integration for Chronos-Agent.
Demonstrates wrapping OpenAI Function Calling tool executions with ChronosAgentWrapper.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from chronos import ChronosHarness, ChronosAgentWrapper
from .tools import tool_read_orders, tool_process_refund, tool_write_audit_script

OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_orders",
            "description": "Look up order details by Order ID from the orders database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The 4-digit Order ID (e.g. '1050')"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Processes a refund for an order and updates the company financial ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID to refund"},
                    "amount": {"type": "number", "description": "Dollar amount to refund"},
                    "reason": {"type": "string", "description": "Reason for refund"},
                    "simulate_crash": {"type": "boolean", "description": "Set to True to simulate a database crash during write"}
                },
                "required": ["order_id", "amount", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_audit_script",
            "description": "Writes a Python audit verification script to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Target filename (e.g. 'audit_1050.py')"},
                    "code_content": {"type": "string", "description": "Python source code for the audit script"}
                },
                "required": ["filename", "code_content"]
            }
        }
    }
]


class OpenAIBusinessAgent:
    """
    OpenAI-powered agent wrapper protected by Chronos.
    Supports real OpenAI API calls when OPENAI_API_KEY is present in env,
    or a high-fidelity synthetic LLM turn simulation fallback.
    """
    def __init__(self, harness: ChronosHarness):
        self.harness = harness
        self.wrapper = ChronosAgentWrapper(harness)
        # Load OPENAI_API_KEY from environment or project .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        self.api_key = line.split("=", 1)[1].strip("'\" \t")
                        os.environ["OPENAI_API_KEY"] = self.api_key
                        break

        # Register raw tools wrapped with Chronos
        self.tools_map: Dict[str, Callable] = {
            "read_orders": self.wrapper.wrap_tool(
                "read_orders",
                lambda args: tool_read_orders(args, harness.workspace_dir)
            ),
            "process_refund": self.wrapper.wrap_tool(
                "process_refund",
                lambda args: tool_process_refund(args, harness.workspace_dir)
            ),
            "write_audit_script": self.wrapper.wrap_tool(
                "write_audit_script",
                lambda args: tool_write_audit_script(args, harness.workspace_dir)
            ),
        }

    def run_turn(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        prompt_stack: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes an OpenAI tool call within Chronos transactional protection.
        1. Takes Dual-Ledger snapshot.
        2. Enforces Pre-flight Invariants & Role Policies.
        3. Reverts filesystem and injects diagnostic hint on error.
        """
        if tool_name not in self.tools_map:
            raise ValueError(f"Unknown tool: '{tool_name}'")

        # Execute through Chronos wrapper
        chronos_fn = self.tools_map[tool_name]
        result = chronos_fn(tool_args, prompt_stack)
        return result

    def execute_with_openai_llm(
        self,
        user_request: str,
        target_order_id: str,
        requested_refund_amount: float,
        simulate_crash: bool = False
    ) -> Dict[str, Any]:
        """
        Demonstrates OpenAI Function Calling loop protected by Chronos.
        """
        prompt_stack = [
            {"role": "system", "content": "You are an E-Commerce Financial Agent authorized to inspect orders and process refunds."},
            {"role": "user", "content": user_request}
        ]

        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                # Call OpenAI ChatCompletions
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=prompt_stack,
                    tools=OPENAI_TOOL_SCHEMAS,
                    tool_choice="auto"
                )
                choice = response.choices[0].message
                if choice.tool_calls:
                    tc = choice.tool_calls[0]
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    if simulate_crash:
                        fn_args["simulate_crash"] = True

                    return self.run_turn(fn_name, fn_args, prompt_stack)
            except Exception as ex:
                print(f"[OpenAI Client Note] Falling back to synthetic LLM runner: {ex}")

        # Fallback / Direct execution simulation for zero-dependency local testing
        fn_args = {
            "order_id": target_order_id,
            "amount": requested_refund_amount,
            "reason": "Customer request processed via LLM pipeline",
            "simulate_crash": simulate_crash
        }
        return self.run_turn("process_refund", fn_args, prompt_stack)
