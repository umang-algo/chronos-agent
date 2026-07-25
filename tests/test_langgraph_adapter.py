import tempfile
import unittest
from pathlib import Path
from chronos.core import ChronosHarness
from chronos.adapters.langgraph import ChronosLangGraphMiddleware

class TestLangGraphAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="langgraph_chronos_")
        self.harness = ChronosHarness(self.temp_dir)
        self.middleware = ChronosLangGraphMiddleware(self.harness)
        self.test_file = Path(self.temp_dir) / "config.py"
        self.test_file.write_text("DEBUG = False\n")

    def tearDown(self):
        self.harness.cleanup()

    def test_langgraph_node_failure_and_rollback(self):
        # Define a node function that corrupts workspace and crashes
        def crashing_node(state):
            self.test_file.write_text("DEBUG = 'CORRUPTED'")
            raise ValueError("Invalid configuration schema")

        wrapped_node = self.middleware.wrap_node("config_node", crashing_node)

        initial_state = {
            "messages": [{"role": "user", "content": "Set DEBUG mode to True"}],
            "step": 1
        }

        # Run wrapped LangGraph node
        new_state = wrapped_node(initial_state)

        # 1. Assert rollback occurred flag set
        self.assertTrue(new_state.get("chronos_rollback_occurred"))

        # 2. Assert Workspace file restored to original clean state
        self.assertEqual(self.test_file.read_text(), "DEBUG = False\n")

        # 3. Assert system prompt hint injected into messages
        messages = new_state["messages"]
        self.assertEqual(messages[-1]["role"], "system")
        self.assertIn("Invalid configuration schema", messages[-1]["content"])

if __name__ == "__main__":
    unittest.main()
