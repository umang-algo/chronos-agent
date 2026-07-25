"""
Interactive Terminal CLI Runner for Chronos-Agent.
Enables real-time prompt testing in the terminal with live display of:
1. Disk File State Diffs
2. Dual-Ledger Checkpoints
3. Causal Execution Trace Graph (CTG) Failure Lineage
4. Counterfactual Diagnostic System Prompt Injections
"""

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chronos import ChronosHarness, PolicyEngine, Policy, ChronosAgentWrapper
from examples.langgraph_business_demo.policy_rules import create_refund_agent_policy, check_refund_invariant
from examples.langgraph_business_demo.tools import tool_read_orders, tool_process_refund, tool_write_audit_script
from examples.langgraph_business_demo.generate_large_db import generate_1000_orders

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title: str):
    border = "═" * 76
    print(f"\n{CYAN}{BOLD}╔{border}╗")
    print(f"║ {title.center(74)} ║")
    print(f"╚{border}╝{RESET}\n")

def print_box(title: str, text: str, color: str = CYAN):
    border = "─" * 76
    print(f"{color}{BOLD}┌{border}┐")
    print(f"│ {title.ljust(72)} │")
    print(f"├{border}┤{RESET}")
    for line in text.split("\n"):
        print(f"{color}│ {line.ljust(72)} │{RESET}")
    print(f"{color}{BOLD}└{border}┘{RESET}")


def run_interactive_cli():
    print_header("CHRONOS-AGENT INTERACTIVE TERMINAL TESTER")
    print(f"{CYAN}Type your test prompt or try one of the following built-in commands:{RESET}")
    print(f"  {BOLD}1{RESET} : Process valid $25.00 refund on Order #1050")
    print(f"  {BOLD}2{RESET} : Attempt illegal $999,999.00 refund on Order #1050 (Test Invariant Guard)")
    print(f"  {BOLD}3{RESET} : Write broken Python code (Test Pre-flight AST Syntax Guard)")
    print(f"  {BOLD}4{RESET} : Simulate database crash mid-write (Test Time-Travel Rollback)")
    print(f"  {BOLD}dag{RESET} : Print current Causal Trace Graph (CTG) Mermaid Diagram")
    print(f"  {BOLD}exit{RESET}: Exit interactive tester\n")

    with tempfile.TemporaryDirectory(prefix="chronos_cli_ws_") as temp_ws:
        ws_path = Path(temp_ws)
        db_dir = ws_path / "mock_db"

        # Generate 1,000 orders
        print(f"{MAGENTA}Initializing workspace with 1,000 sample orders...{RESET}")
        generate_1000_orders(str(db_dir))

        harness = ChronosHarness(str(ws_path))
        policy_engine = create_refund_agent_policy()

        harness.verifier.register_pre_flight_rule(
            "role_policy_guard",
            policy_engine.build_verifier_rule("refund_agent")
        )
        harness.verifier.register_pre_flight_rule(
            "financial_invariant_guard",
            check_refund_invariant
        )

        wrapper = ChronosAgentWrapper(harness)

        # Setup wrapped tools
        def _wrap(tool_name, tool_fn):
            raw_wrapped = wrapper.wrap_tool(tool_name, tool_fn)
            def _executor(args, p_stack):
                args_with_ws = {**args, "workspace_dir": harness.workspace_dir}
                return raw_wrapped(args_with_ws, p_stack)
            return _executor

        wrapped_refund = _wrap("process_refund", lambda args: tool_process_refund(args, harness.workspace_dir))
        wrapped_write = _wrap("write_file", lambda args: tool_write_audit_script({"filename": args.get("filepath", "script.py"), "code_content": args.get("content", ""), "workspace_dir": harness.workspace_dir}, harness.workspace_dir))

        prompt_stack = [{"role": "system", "content": "You are an E-Commerce Agent protected by Chronos."}]

        while True:
            try:
                user_input = input(f"\n{BOLD}{CYAN}Chronos> {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive tester.")
                break

            if not user_input or user_input.lower() == "exit":
                print("Exiting interactive tester.")
                break

            if user_input.lower() == "dag":
                print_box("CAUSAL TRACE GRAPH (MERMAID DIAGRAM CODE)", harness.causal_graph.to_mermaid(), MAGENTA)
                continue

            prompt_stack.append({"role": "user", "content": user_input})

            # Map user inputs to actions
            if user_input == "1" or "valid" in user_input.lower():
                print_box("STEP: VALID REFUND ACTION", "Processing $25.00 refund for Order #1050", GREEN)
                tool_args = {"order_id": "1050", "amount": 25.0, "reason": "Valid refund turn"}
                res = wrapped_refund(tool_args, prompt_stack)
            elif user_input == "2" or "excessive" in user_input.lower() or "illegal" in user_input.lower():
                print_box("STEP: ILLEGAL REFUND ACTION", "Attempting $999,999.00 refund for Order #1050", YELLOW)
                tool_args = {"order_id": "1050", "amount": 999999.0, "reason": "Illegal excessive refund"}
                res = wrapped_refund(tool_args, prompt_stack)
            elif user_input == "3" or "syntax" in user_input.lower():
                print_box("STEP: WRITE BROKEN PYTHON", "Attempting to write code with syntax error", YELLOW)
                bad_content = "def calculate_tax(amount -> float:\n    return amount * 0.08"
                tool_args = {"filepath": "broken_script.py", "content": bad_content}
                res = wrapped_write(tool_args, prompt_stack)
            elif user_input == "4" or "crash" in user_input.lower():
                print_box("STEP: SIMULATE DB CRASH", "Tool throws exception mid-write", RED)
                tool_args = {"order_id": "1050", "amount": 15.0, "simulate_crash": True}
                res = wrapped_refund(tool_args, prompt_stack)
            else:
                print(f"{YELLOW}Processing turn for custom prompt: '{user_input}'...{RESET}")
                tool_args = {"order_id": "1050", "amount": 25.0, "reason": user_input}
                res = wrapped_refund(tool_args, prompt_stack)

            # Display execution outcome
            if res["success"]:
                print(f"{GREEN}✓ ACTION COMMITTED! Checkpoint Saved: {res['checkpoint'].checkpoint_id}{RESET}")
                print(f"{GREEN}  Result: {res['result']['message']}{RESET}")
            else:
                print(f"{RED}🛡️ CHRONOS PRE-FLIGHT GUARD INTERCEPTION / ROLLBACK EXECUTED!{RESET}")
                print(f"{YELLOW}  Diagnostic Hint: {res['hint']}{RESET}")

            # Display file state on disk
            ledger_path = ws_path / "mock_db" / "financial_ledger.json"
            print(f"\n{CYAN}Current Disk File State (mock_db/financial_ledger.json):{RESET}")
            print(f"{YELLOW}{ledger_path.read_text().strip()}{RESET}\n")

        harness.cleanup()

if __name__ == "__main__":
    run_interactive_cli()
