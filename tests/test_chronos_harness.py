import tempfile
import unittest
from pathlib import Path
from chronos.core import ChronosHarness

class TestChronosHarness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_harness_")
        self.harness = ChronosHarness(self.temp_dir)
        self.test_file = Path(self.temp_dir) / "app.py"
        self.test_file.write_text("# Initial clean code\ndef calculate():\n    return 10\n")

    def tearDown(self):
        self.harness.cleanup()

    def test_pre_flight_syntax_error_rejection(self):
        prompt_stack = [{"role": "user", "content": "Write a function to process payments"}]
        cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        def write_file_tool(args):
            filepath = Path(self.temp_dir) / args["filepath"]
            filepath.write_text(args["content"])
            return "File written"

        # Attempt to write invalid syntax Python code
        invalid_payload = {
            "filepath": "app.py",
            "content": "def calculate(:\n    return 10"  # Invalid syntax missing closing paren
        }

        success, result, hint = self.harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args=invalid_payload,
            tool_fn=write_file_tool,
            prompt_stack=prompt_stack,
            current_checkpoint=cp
        )

        self.assertFalse(success)
        self.assertIn("Python SyntaxError", hint)
        # Check that target file remains untouched on disk!
        self.assertEqual(self.test_file.read_text(), "# Initial clean code\ndef calculate():\n    return 10\n")

    def test_time_travel_rollback_on_tool_failure(self):
        prompt_stack = [{"role": "user", "content": "Refactor codebase"}]
        cp1 = self.harness.create_checkpoint(prompt_stack=prompt_stack)

        # Modify file with valid syntax, but tool execution logic fails
        self.test_file.write_text("# Modified broken version\ndef calculate():\n    raise ValueError('Database connection lost')\n")

        # Perform rollback
        restored_prompts = self.harness.rollback(
            target_checkpoint_id=cp1.checkpoint_id,
            reason="Integration test failed",
            counterfactual_hint="Do not modify calculate() signature."
        )

        # Assert filesystem restored to original clean state
        self.assertEqual(self.test_file.read_text(), "# Initial clean code\ndef calculate():\n    return 10\n")
        
        # Assert counterfactual hint injected into restored prompt stack
        self.assertEqual(len(restored_prompts), 2)
        self.assertEqual(restored_prompts[1]["role"], "system")
        self.assertIn("Do not modify calculate() signature", restored_prompts[1]["content"])

if __name__ == "__main__":
    unittest.main()
