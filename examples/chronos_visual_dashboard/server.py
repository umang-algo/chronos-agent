"""
Chronos-Agent Visual Dashboard & Real Conversational Agent Server.

Provides a live web UI for:
1. Real Conversational AI Agent (OpenAI gpt-4o-mini) with function calling.
2. Real-time ChronosHarness Storage & Dual-Ledger Checkpoint Inspector.
3. Live Causal Trace Graph (CTG) DAG Visualizer & Failure Diagnostic Stream.
4. Real-time ShadowFS Workspace File Explorer.

Usage:
  python3 examples/chronos_visual_dashboard/server.py [port]
"""

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chronos import ChronosHarness, PolicyEngine, Policy, ChronosAgentWrapper
from examples.langgraph_business_demo.policy_rules import create_refund_agent_policy, check_refund_invariant
from examples.langgraph_business_demo.tools import tool_read_orders, tool_process_refund, tool_write_audit_script
from examples.langgraph_business_demo.generate_large_db import generate_1000_orders
from examples.langgraph_business_demo.openai_runner import OPENAI_TOOL_SCHEMAS

PORT = 8080
STATIC_DIR = Path(__file__).parent / "static"

# Global state for dashboard demo
DEMO_WORKSPACE = None
DEMO_HARNESS = None
DEMO_POLICY = None
PROMPT_STACK: List[Dict[str, Any]] = []


def load_env_api_key():
    """Auto-loads OPENAI_API_KEY from environment or project .env file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip("'\" \t")
                    os.environ["OPENAI_API_KEY"] = api_key
                    break
    return api_key


def init_demo_state():
    global DEMO_WORKSPACE, DEMO_HARNESS, DEMO_POLICY, PROMPT_STACK
    if DEMO_HARNESS:
        try:
            DEMO_HARNESS.cleanup()
        except Exception:
            pass

    load_env_api_key()

    DEMO_WORKSPACE = tempfile.mkdtemp(prefix="chronos_chat_ui_ws_")
    db_dir = Path(DEMO_WORKSPACE) / "mock_db"
    generate_1000_orders(str(db_dir))

    DEMO_HARNESS = ChronosHarness(DEMO_WORKSPACE)
    DEMO_POLICY = create_refund_agent_policy()

    DEMO_HARNESS.verifier.register_pre_flight_rule(
        "role_policy_guard",
        DEMO_POLICY.build_verifier_rule("refund_agent")
    )
    DEMO_HARNESS.verifier.register_pre_flight_rule(
        "financial_invariant_guard",
        check_refund_invariant
    )

    PROMPT_STACK = [
        {
            "role": "system",
            "content": (
                "You are an E-Commerce Customer Support & Financial Audit AI Agent protected by Chronos-Agent.\n"
                "You have access to tools: 'read_orders', 'process_refund', and 'write_audit_script'.\n"
                "When requested to refund or inspect orders, call the appropriate tool."
            )
        }
    ]


class ChronosDashboardHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving dashboard UI and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json_response(self.get_system_status())
        elif path == "/api/ctg-dag":
            self.send_json_response(self.get_ctg_dag_data())
        elif path == "/api/checkpoints":
            self.send_json_response(self.get_checkpoints_data())
        elif path == "/api/files":
            self.send_json_response(self.get_files_tree())
        elif path == "/api/reset":
            init_demo_state()
            self.send_json_response({"status": "reset", "message": "Workspace reset to 1,000 orders clean state."})
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/chat":
            self.send_json_response(self.handle_chat_turn(payload))
        elif path == "/api/file-content":
            self.send_json_response(self.handle_file_content(payload))
        elif path == "/api/execute-action":
            self.send_json_response(self.handle_execute_action(payload))
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    # ------------------------------------------------------------------
    # API Handlers
    # ------------------------------------------------------------------

    def get_system_status(self) -> Dict[str, Any]:
        harness = DEMO_HARNESS
        api_key = os.getenv("OPENAI_API_KEY")
        return {
            "current_step": harness.current_step,
            "checkpoints_count": len(harness.ledger.checkpoints),
            "ctg_nodes_count": len(harness.causal_graph.nodes),
            "ctg_edges_count": len(harness.causal_graph.edges),
            "failures_count": len(harness.causal_graph.find_all_failures()),
            "workspace_dir": harness.workspace_dir,
            "has_openai_key": bool(api_key),
            "model_name": "gpt-4o-mini" if api_key else "Synthetic LLM Runner (Set OPENAI_API_KEY)",
        }

    def get_ctg_dag_data(self) -> Dict[str, Any]:
        harness = DEMO_HARNESS
        nodes = [n.to_dict() for n in harness.causal_graph.nodes.values()]
        edges = [e.to_dict() for e in harness.causal_graph.edges]
        mermaid_code = harness.causal_graph.to_mermaid()

        failures = harness.causal_graph.find_all_failures()
        explanation = ""
        if failures:
            explanation = harness.causal_graph.explain_failure(failures[-1].node_id)

        return {
            "nodes": nodes,
            "edges": edges,
            "mermaid": mermaid_code,
            "latest_failure_explanation": explanation,
        }

    def get_checkpoints_data(self) -> Dict[str, Any]:
        harness = DEMO_HARNESS
        checkpoints_list = []
        for cp in harness.ledger.checkpoints.values():
            checkpoints_list.append({
                "checkpoint_id": cp.checkpoint_id,
                "step_number": cp.step_number,
                "created_at": cp.created_at,
                "fs_snapshot_id": cp.fs_snapshot_id,
                "tool_calls": cp.turn.tool_calls,
                "prompt_snapshot_count": len(cp.turn.prompt_snapshot),
            })
        return {"checkpoints": sorted(checkpoints_list, key=lambda x: x["step_number"], reverse=True)}

    def get_files_tree(self) -> Dict[str, Any]:
        ws_path = Path(DEMO_HARNESS.workspace_dir)
        files = []
        for root, dirs, filenames in os.walk(ws_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in filenames:
                if not f.endswith(".pyc"):
                    abs_p = Path(root) / f
                    rel_p = str(abs_p.relative_to(ws_path))
                    files.append({
                        "rel_path": rel_p,
                        "size_bytes": abs_p.stat().st_size if abs_p.exists() else 0
                    })
        return {"files": sorted(files, key=lambda x: x["rel_path"])}

    def handle_file_content(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rel_path = payload.get("rel_path", "mock_db/financial_ledger.json")
        ws_path = Path(DEMO_HARNESS.workspace_dir)
        file_path = ws_path / rel_path
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text()
                return {"success": True, "rel_path": rel_path, "content": content}
            except Exception as ex:
                return {"success": False, "error": str(ex)}
        return {"success": False, "error": "File not found"}

    def handle_chat_turn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a real conversational turn:
        1. Appends user message to PROMPT_STACK.
        2. Calls OpenAI API (or fallback tool router).
        3. Wraps tool execution with ChronosAgentWrapper.
        4. Handles pre-flight invariant rejections & self-correction turns.
        """
        user_message = payload.get("message", "").strip()
        if not user_message:
            return {"error": "Empty message"}

        PROMPT_STACK.append({"role": "user", "content": user_message})
        harness = DEMO_HARNESS
        wrapper = ChronosAgentWrapper(harness)

        # Setup wrapped tools (inject workspace_dir into tool_args for verifiers)
        def _wrap(tool_name, tool_fn):
            raw_wrapped = wrapper.wrap_tool(tool_name, tool_fn)
            def _executor(args, p_stack):
                args_with_ws = {**args, "workspace_dir": harness.workspace_dir}
                return raw_wrapped(args_with_ws, p_stack)
            return _executor

        wrapped_tools = {
            "read_orders": _wrap(
                "read_orders",
                lambda args: tool_read_orders(args, harness.workspace_dir)
            ),
            "process_refund": _wrap(
                "process_refund",
                lambda args: tool_process_refund(args, harness.workspace_dir)
            ),
            "write_audit_script": _wrap(
                "write_audit_script",
                lambda args: tool_write_audit_script(args, harness.workspace_dir)
            ),
            "write_file": _wrap(
                "write_file",
                lambda args: tool_write_audit_script({"filename": args.get("filepath", "script.py"), "code_content": args.get("content", ""), "workspace_dir": harness.workspace_dir}, harness.workspace_dir)
            ),
        }

        events = []
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=PROMPT_STACK,
                    tools=OPENAI_TOOL_SCHEMAS,
                    tool_choice="auto"
                )
                choice = response.choices[0].message

                if choice.tool_calls:
                    for tc in choice.tool_calls:
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except Exception:
                            fn_args = {}

                        events.append({"type": "tool_call", "name": fn_name, "args": fn_args})

                        if fn_name in wrapped_tools:
                            res = wrapped_tools[fn_name](fn_args, PROMPT_STACK)
                            events.append({"type": "tool_result", "name": fn_name, "success": res["success"], "result": res["result"], "hint": res["hint"]})

                            if not res["success"]:
                                # Feed counterfactual hint back to LLM for self-correction
                                PROMPT_STACK.append({
                                    "role": "system",
                                    "content": f"⚠️ Chronos Invariant Interception: {res['hint']}"
                                })
                                # Self-correction follow-up turn
                                retry_resp = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=PROMPT_STACK
                                )
                                self_correction = retry_resp.choices[0].message.content
                                PROMPT_STACK.append({"role": "assistant", "content": self_correction})
                                return {
                                    "user_message": user_message,
                                    "assistant_response": self_correction,
                                    "events": events,
                                    "self_corrected": True
                                }
                            else:
                                PROMPT_STACK.append({"role": "function", "name": fn_name, "content": json.dumps(res["result"])})

                assistant_msg = choice.content or "Action completed."
                PROMPT_STACK.append({"role": "assistant", "content": assistant_msg})
                return {
                    "user_message": user_message,
                    "assistant_response": assistant_msg,
                    "events": events,
                    "self_corrected": False
                }

            except Exception as ex:
                events.append({"type": "openai_note", "note": f"OpenAI API Note: {ex}. Using synthetic agent execution fallback."})

        # Synthetic Fallback Router for local deterministic test prompts
        msg_lower = user_message.lower()
        if "refund" in msg_lower or "process" in msg_lower:
            # Extract order ID if present or default
            order_id = "1050"
            for word in user_message.split():
                clean_word = word.strip("#,.")
                if clean_word.isdigit() and len(clean_word) >= 4:
                    order_id = clean_word

            amount = 25.0
            if "$999" in user_message or "5000" in user_message or "excessive" in msg_lower:
                amount = 999999.0
            elif "$" in user_message:
                try:
                    amount = float(user_message.split("$")[1].split()[0].strip(",."))
                except Exception:
                    pass

            fn_args = {"order_id": order_id, "amount": amount, "reason": user_message}
            events.append({"type": "tool_call", "name": "process_refund", "args": fn_args})

            res = wrapped_tools["process_refund"](fn_args, PROMPT_STACK)
            events.append({"type": "tool_result", "name": "process_refund", "success": res["success"], "result": res["result"], "hint": res["hint"]})

            if res["success"]:
                reply = f"Processed ${amount:.2f} refund for Order #{order_id} cleanly via Chronos Substrate."
            else:
                reply = f"🚫 Action Intercepted by Chronos: {res['hint']}"

            PROMPT_STACK.append({"role": "assistant", "content": reply})
            return {"user_message": user_message, "assistant_response": reply, "events": events, "self_corrected": not res["success"]}

        elif "python" in msg_lower or "syntax" in msg_lower or "script" in msg_lower:
            bad_content = "def calculate_tax(amount -> float:\n    return amount * 0.08"
            fn_args = {"filepath": "audit_script.py", "content": bad_content}
            events.append({"type": "tool_call", "name": "write_file", "args": fn_args})

            res = wrapped_tools["write_file"](fn_args, PROMPT_STACK)
            events.append({"type": "tool_result", "name": "write_file", "success": res["success"], "result": res["result"], "hint": res["hint"]})

            reply = f"🚫 Pre-flight AST Guard REJECTED broken syntax: {res['hint']}"
            PROMPT_STACK.append({"role": "assistant", "content": reply})
            return {"user_message": user_message, "assistant_response": reply, "events": events, "self_corrected": True}

        else:
            # Default response
            reply = f"I received your instruction: '{user_message}'. Chronos checkpoint created for step {harness.current_step}."
            PROMPT_STACK.append({"role": "assistant", "content": reply})
            return {"user_message": user_message, "assistant_response": reply, "events": events, "self_corrected": False}

    def handle_execute_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action_type = payload.get("action_type", "valid_refund")
        order_id = str(payload.get("order_id", "1050"))
        amount = float(payload.get("amount", 25.0))
        harness = DEMO_HARNESS

        prompt_stack = [{"role": "user", "content": f"Action: {action_type} for order #{order_id}"}]
        cp = harness.create_checkpoint(prompt_stack=prompt_stack)
        t0 = time.time()

        if action_type == "valid_refund":
            tool_args = {"order_id": order_id, "amount": amount, "reason": "Dashboard UI turn", "workspace_dir": harness.workspace_dir}
            def tool_fn(args):
                return tool_process_refund(tool_args, harness.workspace_dir)
            success, result, hint = harness.execute_tool_transactional(
                tool_name="process_refund",
                tool_args=tool_args,
                tool_fn=tool_fn,
                prompt_stack=prompt_stack,
                current_checkpoint=cp
            )
            elapsed_ms = (time.time() - t0) * 1000
            return {"action": "valid_refund", "success": success, "result": result, "hint": hint, "elapsed_ms": elapsed_ms, "checkpoint_id": cp.checkpoint_id}

        elif action_type == "invariant_violation":
            tool_args = {"order_id": order_id, "amount": 999999.00, "reason": "Illegal amount", "workspace_dir": harness.workspace_dir}
            def tool_fn(args):
                return tool_process_refund(tool_args, harness.workspace_dir)
            success, result, hint = harness.execute_tool_transactional(
                tool_name="process_refund",
                tool_args=tool_args,
                tool_fn=tool_fn,
                prompt_stack=prompt_stack,
                current_checkpoint=cp
            )
            elapsed_ms = (time.time() - t0) * 1000
            return {"action": "invariant_violation", "success": success, "result": result, "hint": hint, "elapsed_ms": elapsed_ms, "checkpoint_id": cp.checkpoint_id}

        elif action_type == "syntax_error":
            bad_content = "def calculate_tax(amount -> float:\n    return amount * 0.08"
            tool_args = {"filepath": "broken_script.py", "content": bad_content, "workspace_dir": harness.workspace_dir}
            def tool_fn(args):
                return tool_write_audit_script({"filename": "broken_script.py", "code_content": bad_content}, harness.workspace_dir)
            success, result, hint = harness.execute_tool_transactional(
                tool_name="write_file",
                tool_args=tool_args,
                tool_fn=tool_fn,
                prompt_stack=prompt_stack,
                current_checkpoint=cp
            )
            elapsed_ms = (time.time() - t0) * 1000
            return {"action": "syntax_error", "success": success, "result": result, "hint": hint, "elapsed_ms": elapsed_ms, "checkpoint_id": cp.checkpoint_id}

        elif action_type == "database_crash":
            tool_args = {"order_id": order_id, "amount": 15.00, "simulate_crash": True, "workspace_dir": harness.workspace_dir}
            def tool_fn(args):
                return tool_process_refund(tool_args, harness.workspace_dir)
            success, result, hint = harness.execute_tool_transactional(
                tool_name="process_refund",
                tool_args=tool_args,
                tool_fn=tool_fn,
                prompt_stack=prompt_stack,
                current_checkpoint=cp
            )
            elapsed_ms = (time.time() - t0) * 1000
            return {"action": "database_crash", "success": success, "result": result, "hint": hint, "elapsed_ms": elapsed_ms, "checkpoint_id": cp.checkpoint_id}

        return {"error": f"Unknown action: '{action_type}'"}


def main():
    init_demo_state()
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server_address = ("", port)
    httpd = HTTPServer(server_address, ChronosDashboardHandler)

    print(f"\n\033[96m\033[1m" + "═" * 76)
    print(f" 🚀 CHRONOS-AGENT REAL CONVERSATIONAL AGENT UI SERVER RUNNING")
    print(f" 🌐 Dashboard URL: \033[92mhttp://localhost:{port}\033[96m")
    print(f" 🔑 OpenAI API Model: \033[93m{'gpt-4o-mini (Active)' if os.getenv('OPENAI_API_KEY') else 'Synthetic LLM Runner'}\033[96m")
    print(f" 📁 Managed Workspace: {DEMO_WORKSPACE}")
    print("═" * 76 + "\033[0m\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\033[93mShutting down Chronos Dashboard Server...\033[0m")
        if DEMO_HARNESS:
            DEMO_HARNESS.cleanup()
        httpd.server_close()


if __name__ == "__main__":
    main()
