import tempfile
import unittest
from pathlib import Path
from chronos.policy import PolicyEngine, Policy, PolicyViolation
from chronos.core import ChronosHarness

class TestPolicyEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PolicyEngine()

    # ------------------------------------------------------------------
    # Unit Tests — PolicyEngine standalone
    # ------------------------------------------------------------------

    def test_no_policy_blocks_all_actions(self):
        """No registered policy = default deny everything."""
        violations = self.engine.enforce(
            role="unknown_agent",
            tool_name="write_file",
            tool_args={"filepath": "app.py", "content": "..."}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation, "NO_POLICY")

    def test_denied_tool_blocked(self):
        self.engine.register_policy(Policy(
            role="finance_agent",
            allow_tools=["read_file", "write_report"],
            deny_tools=["execute_bash", "delete_file"],
            allow_all_tools=False
        ))
        violations = self.engine.enforce(
            role="finance_agent",
            tool_name="execute_bash",
            tool_args={"command": "ls -la"}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation, "TOOL_DENIED")

    def test_allowed_tool_passes(self):
        self.engine.register_policy(Policy(
            role="finance_agent",
            allow_tools=["read_file", "write_report"],
            deny_tools=["execute_bash"],
        ))
        violations = self.engine.enforce(
            role="finance_agent",
            tool_name="read_file",
            tool_args={"filepath": "reports/q1.pdf"}
        )
        # No violations if tool is allowed (read scope not set = open)
        self.assertEqual(len(violations), 0)

    def test_tool_not_in_allow_list_blocked(self):
        self.engine.register_policy(Policy(
            role="read_only_agent",
            allow_tools=["read_file"],
        ))
        violations = self.engine.enforce(
            role="read_only_agent",
            tool_name="write_file",
            tool_args={"filepath": "app.py", "content": "..."}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation, "TOOL_NOT_ALLOWED")

    def test_write_scope_violation(self):
        self.engine.register_policy(Policy(
            role="report_agent",
            allow_tools=["write_file"],
            allow_write=["reports/*"],
        ))
        violations = self.engine.enforce(
            role="report_agent",
            tool_name="write_file",
            tool_args={"filepath": "ledger/private.csv"}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation, "WRITE_SCOPE_VIOLATION")

    def test_write_within_scope_passes(self):
        self.engine.register_policy(Policy(
            role="report_agent",
            allow_tools=["write_file"],
            allow_write=["reports/*"],
        ))
        violations = self.engine.enforce(
            role="report_agent",
            tool_name="write_file",
            tool_args={"filepath": "reports/q2_summary.md"}
        )
        self.assertEqual(len(violations), 0)

    def test_read_scope_violation(self):
        self.engine.register_policy(Policy(
            role="public_agent",
            allow_tools=["read_file"],
            allow_read=["public/*"],
        ))
        violations = self.engine.enforce(
            role="public_agent",
            tool_name="read_file",
            tool_args={"filepath": "internal/secrets.env"}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation, "READ_SCOPE_VIOLATION")

    def test_allow_all_tools_except_denied(self):
        self.engine.register_policy(Policy(
            role="superagent",
            allow_all_tools=True,
            deny_tools=["nuke_database"],
        ))
        # Any allowed tool passes with no path-scope set
        ok = self.engine.enforce("superagent", "write_file", {})
        self.assertEqual(len(ok), 0)
        # Denied tool still blocked
        blocked = self.engine.enforce("superagent", "nuke_database", {})
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].violation, "TOOL_DENIED")

    # ------------------------------------------------------------------
    # Integration Tests — Policy wired into ChronosHarness verifier
    # ------------------------------------------------------------------

    def test_policy_wired_into_harness_blocks_tool(self):
        temp_dir = tempfile.mkdtemp(prefix="policy_harness_")
        harness = ChronosHarness(temp_dir)

        engine = PolicyEngine()
        engine.register_policy(Policy(
            role="restricted_agent",
            allow_tools=["read_file"],
            deny_tools=["execute_bash"],
        ))

        # Wire policy into the harness verifier
        harness.verifier.register_pre_flight_rule(
            "policy_engine",
            engine.build_verifier_rule("restricted_agent")
        )

        # Create a test file and checkpoint
        test_file = Path(temp_dir) / "app.py"
        test_file.write_text("def main(): pass\n")
        prompt_stack = [{"role": "user", "content": "Run bash command"}]
        cp = harness.create_checkpoint(prompt_stack=prompt_stack)

        # Try to run an execute_bash tool (should be blocked by policy)
        success, result, hint = harness.execute_tool_transactional(
            tool_name="execute_bash",
            tool_args={"command": "ls -la"},
            tool_fn=lambda args: "executed",
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        harness.cleanup()

        self.assertFalse(success)
        self.assertIsNotNone(hint)
        self.assertIn("execute_bash", hint.lower())

    def test_policy_wired_into_harness_allows_permitted_tool(self):
        temp_dir = tempfile.mkdtemp(prefix="policy_allow_")
        harness = ChronosHarness(temp_dir)

        engine = PolicyEngine()
        engine.register_policy(Policy(
            role="read_agent",
            allow_tools=["read_file"],
        ))

        harness.verifier.register_pre_flight_rule(
            "policy_engine",
            engine.build_verifier_rule("read_agent")
        )

        prompt_stack = [{"role": "user", "content": "Read a file"}]
        cp = harness.create_checkpoint(prompt_stack=prompt_stack)

        success, result, hint = harness.execute_tool_transactional(
            tool_name="read_file",
            tool_args={"filepath": "app.py"},
            tool_fn=lambda args: "file_content",
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        harness.cleanup()
        self.assertTrue(success)
        self.assertIsNone(hint)


if __name__ == "__main__":
    unittest.main()
