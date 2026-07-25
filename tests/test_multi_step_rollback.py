import tempfile
import unittest
from pathlib import Path
from chronos.core import ChronosHarness

class TestMultiStepRollback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="multi_step_chronos_")
        self.harness = ChronosHarness(self.temp_dir)
        self.state_file = Path(self.temp_dir) / "counter.txt"
        self.state_file.write_text("0")

    def tearDown(self):
        self.harness.cleanup()

    def test_deep_multi_turn_rollback(self):
        prompt_stack = [{"role": "user", "content": "Increment counter gradually"}]

        # Turn 1: Start Step 1 -> Write 1
        cp1 = self.harness.create_checkpoint(prompt_stack=prompt_stack)
        self.state_file.write_text("1")
        prompt_stack.append({"role": "assistant", "content": "Updated to 1"})

        # Turn 2: Start Step 2 -> Write 2
        cp2 = self.harness.create_checkpoint(prompt_stack=prompt_stack)
        self.state_file.write_text("2")
        prompt_stack.append({"role": "assistant", "content": "Updated to 2"})

        # Turn 3: Start Step 3 -> Write 3
        cp3 = self.harness.create_checkpoint(prompt_stack=prompt_stack)
        self.state_file.write_text("3")
        prompt_stack.append({"role": "assistant", "content": "Updated to 3"})

        # Turn 4: Start Step 4 -> Write CORRUPTED_999
        cp4 = self.harness.create_checkpoint(prompt_stack=prompt_stack)
        self.state_file.write_text("CORRUPTED_999")

        # DEEP ROLLBACK: Rollback to start of Step 3 (which restores state after Step 2 = "2")
        restored_prompts = self.harness.rollback(
            target_checkpoint_id=cp3.checkpoint_id,
            reason="Step 4 caused corruption.",
            counterfactual_hint="Do not exceed counter 2."
        )

        # 1. Assert File restored to start of Step 3 state ("2")
        self.assertEqual(self.state_file.read_text(), "2")

        # 2. Assert Checkpoint 4 was pruned from DualLedger
        self.assertIsNone(self.harness.ledger.get_checkpoint(cp4.checkpoint_id))
        self.assertIsNotNone(self.harness.ledger.get_checkpoint(cp3.checkpoint_id))

        # 3. Assert Restored Prompts contain Counterfactual System Hint
        self.assertEqual(restored_prompts[-1]["role"], "system")
        self.assertIn("Do not exceed counter 2", restored_prompts[-1]["content"])

if __name__ == "__main__":
    unittest.main()
