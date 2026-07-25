"""Tests for Phase 8: Persona Freeze & State Fork."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from chronos.persona import PersonaVault, PersonaManager, FrozenPersona


class TestPersonaFreezeAndFork(unittest.TestCase):

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="persona_ws_")
        self.vault_dir = tempfile.mkdtemp(prefix="persona_vault_")
        Path(self.workspace, "app.py").write_text("def main(): pass\n")
        Path(self.workspace, "config.json").write_text('{"env": "dev"}\n')
        self.vault = PersonaVault(self.vault_dir)
        self.manager = PersonaManager(self.vault)

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)
        shutil.rmtree(self.vault_dir, ignore_errors=True)

    def _freeze(self, label="test_freeze", step=5):
        return self.manager.freeze(
            label=label,
            workspace_dir=self.workspace,
            prompt_stack=[{"role": "user", "content": "Do the task"}],
            step_number=step,
            custom_state={"memory": ["step1", "step2"]},
        )

    # ------------------------------------------------------------------
    # Freeze tests
    # ------------------------------------------------------------------

    def test_freeze_creates_frozen_persona(self):
        frozen = self._freeze()
        self.assertIsInstance(frozen, FrozenPersona)
        self.assertEqual(frozen.label, "test_freeze")
        self.assertEqual(frozen.step_number, 5)
        self.assertEqual(len(frozen.prompt_stack), 1)

    def test_freeze_persists_to_vault(self):
        frozen = self._freeze()
        loaded = self.vault.load(frozen.persona_id)
        self.assertEqual(loaded.persona_id, frozen.persona_id)
        self.assertEqual(loaded.label, frozen.label)

    def test_freeze_copies_workspace_files(self):
        frozen = self._freeze()
        self.assertTrue(os.path.isdir(frozen.workspace_dir))
        files = os.listdir(frozen.workspace_dir)
        self.assertIn("app.py", files)
        self.assertIn("config.json", files)

    def test_freeze_workspace_is_independent_copy(self):
        """Modifying main workspace after freeze must not affect frozen copy."""
        frozen = self._freeze()
        frozen_content = Path(frozen.workspace_dir, "app.py").read_text()
        # Modify main workspace
        Path(self.workspace, "app.py").write_text("MODIFIED CONTENT\n")
        frozen_content_after = Path(frozen.workspace_dir, "app.py").read_text()
        self.assertEqual(frozen_content, frozen_content_after)

    def test_freeze_generates_workspace_hash(self):
        frozen = self._freeze()
        self.assertTrue(len(frozen.workspace_hash) > 0)
        self.assertNotEqual(frozen.workspace_hash, "empty")

    def test_frozen_persona_summary_readable(self):
        frozen = self._freeze(label="my_label")
        summary = frozen.summary()
        self.assertIn("my_label", summary)
        self.assertIn("step=5", summary)

    # ------------------------------------------------------------------
    # Fork tests
    # ------------------------------------------------------------------

    def test_fork_creates_n_isolated_workspaces(self):
        frozen = self._freeze()
        forks = self.manager.fork(frozen, n=3)
        self.assertEqual(len(forks), 3)
        dirs = [f["workspace_dir"] for f in forks]
        # All dirs must be distinct
        self.assertEqual(len(set(dirs)), 3)

    def test_fork_workspaces_are_copies_of_frozen(self):
        frozen = self._freeze()
        forks = self.manager.fork(frozen, n=2)
        for fork in forks:
            self.assertIn("app.py", os.listdir(fork["workspace_dir"]))

    def test_fork_workspaces_are_independent(self):
        """Modifying one fork must not affect other forks."""
        frozen = self._freeze()
        forks = self.manager.fork(frozen, n=2)
        Path(forks[0]["workspace_dir"], "app.py").write_text("FORK_0_MODIFICATION\n")
        content_fork1 = Path(forks[1]["workspace_dir"], "app.py").read_text()
        self.assertNotEqual(content_fork1, "FORK_0_MODIFICATION\n")

    def test_fork_carries_prompt_stack(self):
        frozen = self._freeze()
        forks = self.manager.fork(frozen, n=2)
        for fork in forks:
            self.assertEqual(fork["prompt_stack"], frozen.prompt_stack)

    def test_fork_carries_custom_state(self):
        frozen = self._freeze()
        forks = self.manager.fork(frozen, n=2)
        for fork in forks:
            self.assertEqual(fork["custom_state"], frozen.custom_state)

    # ------------------------------------------------------------------
    # Replay tests
    # ------------------------------------------------------------------

    def test_replay_returns_identical_state(self):
        frozen = self._freeze()
        replay = self.manager.replay(frozen)
        self.assertEqual(replay["persona_id"], frozen.persona_id)
        self.assertEqual(replay["step_number"], frozen.step_number)
        self.assertEqual(replay["prompt_stack"], frozen.prompt_stack)

    def test_replay_workspace_has_original_files(self):
        frozen = self._freeze()
        replay = self.manager.replay(frozen)
        self.assertIn("app.py", os.listdir(replay["workspace_dir"]))

    # ------------------------------------------------------------------
    # Diff tests
    # ------------------------------------------------------------------

    def test_diff_identical_personas_shows_no_changes(self):
        frozen = self._freeze()
        diff = self.manager.diff(frozen, frozen)
        self.assertEqual(len(diff.added_messages), 0)
        self.assertEqual(len(diff.removed_messages), 0)
        self.assertEqual(len(diff.state_changes), 0)
        self.assertEqual(len(diff.files_changed), 0)

    def test_diff_detects_added_messages(self):
        frozen_a = self.manager.freeze(
            label="a", workspace_dir=self.workspace,
            prompt_stack=[{"role": "user", "content": "Step 1"}],
            step_number=1
        )
        frozen_b = self.manager.freeze(
            label="b", workspace_dir=self.workspace,
            prompt_stack=[
                {"role": "user", "content": "Step 1"},
                {"role": "assistant", "content": "Step 2 response"}
            ],
            step_number=2
        )
        diff = self.manager.diff(frozen_a, frozen_b)
        self.assertEqual(len(diff.added_messages), 1)
        self.assertEqual(diff.step_delta, 1)

    def test_diff_detects_state_changes(self):
        frozen_a = self.manager.freeze(
            label="a", workspace_dir=self.workspace,
            prompt_stack=[], step_number=1,
            custom_state={"counter": 0}
        )
        frozen_b = self.manager.freeze(
            label="b", workspace_dir=self.workspace,
            prompt_stack=[], step_number=2,
            custom_state={"counter": 5}
        )
        diff = self.manager.diff(frozen_a, frozen_b)
        self.assertIn("counter", diff.state_changes)
        self.assertEqual(diff.state_changes["counter"], (0, 5))

    def test_diff_summary_readable(self):
        frozen_a = self._freeze(label="a", step=1)
        frozen_b = self._freeze(label="b", step=2)
        diff = self.manager.diff(frozen_a, frozen_b)
        summary = diff.summary()
        self.assertIn("Diff:", summary)

    # ------------------------------------------------------------------
    # Vault list/delete tests
    # ------------------------------------------------------------------

    def test_vault_list_personas(self):
        self._freeze(label="p1")
        self._freeze(label="p2")
        entries = self.vault.list_personas()
        self.assertGreaterEqual(len(entries), 2)

    def test_vault_delete_persona(self):
        frozen = self._freeze()
        self.vault.delete(frozen.persona_id)
        with self.assertRaises(FileNotFoundError):
            self.vault.load(frozen.persona_id)


if __name__ == "__main__":
    unittest.main()
