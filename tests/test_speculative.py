"""Tests for Phase 6: Speculative Parallel Branch Execution."""
import os
import tempfile
import unittest
from pathlib import Path
from chronos.speculative import SpeculativeExecutor, BranchStrategy


class TestSpeculativeExecutor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="spec_test_")
        Path(self.temp_dir, "app.py").write_text("def main(): pass\n")
        Path(self.temp_dir, "config.json").write_text('{"version": 1}\n')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_executor(self):
        return SpeculativeExecutor(main_workspace=self.temp_dir)

    # ------------------------------------------------------------------

    def test_single_branch_success(self):
        """Single branch that always succeeds."""
        executor = self._make_executor()

        def write_good(args, ws_dir):
            Path(ws_dir, "output.txt").write_text("good output")
            return "done"

        result = executor.run_speculative([
            BranchStrategy("strategy_a", "write_file", {}, write_good),
        ])
        self.assertTrue(result.succeeded)
        self.assertEqual(result.winner.branch_name, "strategy_a")

    def test_first_winner_selected_over_later(self):
        """When multiple branches succeed, the first to complete wins."""
        import time
        executor = self._make_executor()

        def fast_write(args, ws_dir):
            Path(ws_dir, "out.txt").write_text("fast")
            return "fast"

        def slow_write(args, ws_dir):
            time.sleep(0.05)
            Path(ws_dir, "out.txt").write_text("slow")
            return "slow"

        result = executor.run_speculative([
            BranchStrategy("fast_branch", "write_file", {}, fast_write),
            BranchStrategy("slow_branch", "write_file", {}, slow_write),
        ])
        self.assertTrue(result.succeeded)
        # fast_branch should win (completes first)
        self.assertEqual(result.winner.branch_name, "fast_branch")

    def test_all_branches_fail_returns_no_winner(self):
        executor = self._make_executor()

        def always_fail(args, ws_dir):
            raise RuntimeError("intentional failure")

        result = executor.run_speculative([
            BranchStrategy("bad_a", "write_file", {}, always_fail),
            BranchStrategy("bad_b", "write_file", {}, always_fail),
        ])
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.winner)
        self.assertEqual(len(result.all_outcomes), 2)

    def test_validator_rejects_bad_branch(self):
        """Validator gates winner selection beyond just no-exception."""
        executor = self._make_executor()

        def write_bad(args, ws_dir):
            Path(ws_dir, "quality.txt").write_text("INVALID")
            return "wrote"

        def write_good(args, ws_dir):
            # Simulate slower but correct branch
            import time; time.sleep(0.02)
            Path(ws_dir, "quality.txt").write_text("VALID")
            return "wrote"

        def validator(ws_dir):
            quality_file = Path(ws_dir, "quality.txt")
            if not quality_file.exists():
                return False, "quality.txt missing"
            content = quality_file.read_text()
            if content == "VALID":
                return True, None
            return False, f"Output was not valid: {content!r}"

        result = executor.run_speculative([
            BranchStrategy("bad_branch",  "write_file", {}, write_bad),
            BranchStrategy("good_branch", "write_file", {}, write_good),
        ], validator=validator)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.winner.branch_name, "good_branch")

    def test_winner_files_merged_into_main_workspace(self):
        """Winner's workspace changes appear in the main workspace after run."""
        executor = self._make_executor()

        def write_marker(args, ws_dir):
            Path(ws_dir, "speculative_output.txt").write_text("WINNER")
            return "ok"

        executor.run_speculative([
            BranchStrategy("winning_branch", "write_file", {}, write_marker),
        ])

        merged_file = Path(self.temp_dir, "speculative_output.txt")
        self.assertTrue(merged_file.exists())
        self.assertEqual(merged_file.read_text(), "WINNER")

    def test_losing_branches_do_not_pollute_workspace(self):
        """Files created only by losing branches must not reach main workspace."""
        executor = self._make_executor()

        def write_loser_file(args, ws_dir):
            Path(ws_dir, "loser_only.txt").write_text("loser")
            raise RuntimeError("loser fails")

        def write_winner_file(args, ws_dir):
            Path(ws_dir, "winner.txt").write_text("winner")
            return "ok"

        executor.run_speculative([
            BranchStrategy("loser",  "write_file", {}, write_loser_file),
            BranchStrategy("winner", "write_file", {}, write_winner_file),
        ])

        self.assertFalse(Path(self.temp_dir, "loser_only.txt").exists())
        self.assertTrue(Path(self.temp_dir, "winner.txt").exists())

    def test_result_summary_non_empty(self):
        executor = self._make_executor()
        result = executor.run_speculative([
            BranchStrategy("b", "write_file", {}, lambda a, ws: "ok"),
        ])
        self.assertIn("Speculative", result.summary())

    def test_elapsed_time_tracked(self):
        executor = self._make_executor()
        result = executor.run_speculative([
            BranchStrategy("b", "write_file", {}, lambda a, ws: "ok"),
        ])
        self.assertGreater(result.total_elapsed_ms, 0)


if __name__ == "__main__":
    unittest.main()
