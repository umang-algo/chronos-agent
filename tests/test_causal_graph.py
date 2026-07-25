import tempfile
import unittest
from pathlib import Path
from chronos.core import ChronosHarness
from chronos.causal_graph import NodeKind, NodeStatus

class TestCausalTraceGraph(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ctg_chronos_")
        self.harness = ChronosHarness(self.temp_dir)
        self.test_file = Path(self.temp_dir) / "app.py"
        self.test_file.write_text("def calculate(x): return x * 10\n")

    def tearDown(self):
        self.harness.cleanup()

    def test_ctg_records_checkpoint_node(self):
        prompt_stack = [{"role": "user", "content": "Refactor app.py"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        ctg = self.harness.causal_graph
        checkpoint_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.CHECKPOINT]
        self.assertEqual(len(checkpoint_nodes), 1)
        self.assertEqual(checkpoint_nodes[0].checkpoint_id, cp.checkpoint_id)

    def test_ctg_records_tool_call_and_failure_chain(self):
        prompt_stack = [{"role": "user", "content": "Write invalid Python"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        bad_content = "def foo(:\n    pass"
        self.harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "app.py", "content": bad_content},
            tool_fn=lambda args: None,
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        ctg = self.harness.causal_graph
        tool_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.TOOL_CALL]
        failure_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.FAILURE]
        invariant_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.INVARIANT_CHECK]

        self.assertGreater(len(tool_nodes), 0, "Expected a TOOL_CALL node")
        self.assertGreater(len(failure_nodes), 0, "Expected a FAILURE node")
        self.assertGreater(len(invariant_nodes), 0, "Expected an INVARIANT_CHECK node")

        # A failure node may be marked REVERTED after rollback — both are valid failure states
        fail_node = failure_nodes[0]
        self.assertIn(fail_node.status, [NodeStatus.FAILURE, NodeStatus.REVERTED])
        self.assertIn("SyntaxError", fail_node.metadata.get("reason", ""))

    def test_ctg_root_cause_trace_is_ordered(self):
        """Root cause trace should walk backwards from failure to origin."""
        prompt_stack = [{"role": "user", "content": "Trigger failure"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        self.harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "app.py", "content": "def bad(:\n    pass"},
            tool_fn=lambda args: None,
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        ctg = self.harness.causal_graph
        failures = ctg.find_all_failures()
        self.assertGreater(len(failures), 0)

        chain = ctg.trace_root_cause(failures[-1].node_id)
        self.assertGreater(len(chain), 1, "Chain should have more than 1 node")
        self.assertEqual(chain[-1].status, NodeStatus.FAILURE)

    def test_ctg_explain_failure_generates_human_readable_text(self):
        prompt_stack = [{"role": "user", "content": "Write bad code"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        self.harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "app.py", "content": "def bad(:\n    pass"},
            tool_fn=lambda args: None,
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        ctg = self.harness.causal_graph
        failures = ctg.find_all_failures()
        explanation = ctg.explain_failure(failures[-1].node_id)

        self.assertIn("Root cause", explanation)
        self.assertIn("Step", explanation)
        self.assertIn("Fix:", explanation)

    def test_ctg_rollback_records_rollback_and_counterfactual_nodes(self):
        prompt_stack = [{"role": "user", "content": "Start task"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        self.harness.rollback(
            target_checkpoint_id=cp.checkpoint_id,
            reason="Something went wrong",
            counterfactual_hint="Try a different approach"
        )

        ctg = self.harness.causal_graph
        rollback_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.ROLLBACK]
        cf_nodes = [n for n in ctg.nodes.values() if n.kind == NodeKind.COUNTERFACTUAL]

        self.assertEqual(len(rollback_nodes), 1)
        self.assertEqual(len(cf_nodes), 1)
        self.assertIn("Try a different approach", cf_nodes[0].metadata.get("hint", ""))

    def test_ctg_to_mermaid_produces_valid_output(self):
        prompt_stack = [{"role": "user", "content": "Test mermaid render"}]
        self.harness.create_checkpoint(prompt_stack=prompt_stack)

        mermaid_output = self.harness.causal_graph.to_mermaid()
        self.assertIn("graph TD", mermaid_output)
        # Node IDs start with 'chk_' for checkpoint nodes
        self.assertIn("chk_", mermaid_output)


if __name__ == "__main__":
    unittest.main()
