"""Tests for Phase 7: JIT Trajectory Compiler."""
import unittest
from chronos.jit_compiler import JITCompiler, TrajectoryStep


class TestJITCompiler(unittest.TestCase):

    def test_no_skill_before_threshold(self):
        jit = JITCompiler(compile_threshold=3)
        for _ in range(2):
            skill = jit.record_step("write_file", {"path": "a.py", "content": "x"}, "ok")
            self.assertIsNone(skill)

    def test_skill_compiled_at_threshold(self):
        jit = JITCompiler(compile_threshold=3)
        skill = None
        for _ in range(3):
            skill = jit.record_step("write_file", {"path": "a.py", "content": "x"}, "ok")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "write_file_compiled")

    def test_match_returns_skill_after_compilation(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("read_file", {"path": "README.md"}, "contents")
        skill = jit.match("read_file", {"path": "README.md"})
        self.assertIsNotNone(skill)

    def test_match_returns_none_for_unknown_tool(self):
        jit = JITCompiler(compile_threshold=2)
        skill = jit.match("unknown_tool", {"x": 1})
        self.assertIsNone(skill)

    def test_compiled_skill_executes_successfully(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("summarize", {"text": "hello world"}, "summary_result")
        skill = jit.match("summarize", {"text": "hello world"})
        self.assertIsNotNone(skill)
        success, result, elapsed_ms = skill.execute({"text": "hello world"})
        self.assertTrue(success)
        self.assertEqual(result, "summary_result")
        self.assertGreaterEqual(elapsed_ms, 0)

    def test_compiled_skill_fails_on_type_mismatch(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("process", {"count": 10}, "result")
        skill = jit.match("process", {"count": 10})
        self.assertIsNotNone(skill)
        # Pass wrong type — should return failure not raise
        success, result, _ = skill.execute({"count": "ten"})
        self.assertFalse(success)

    def test_compiled_skill_fails_on_missing_arg(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("render", {"template": "tmpl", "data": {}}, "html")
        skill = jit.match("render", {"template": "tmpl", "data": {}})
        self.assertIsNotNone(skill)
        success, result, _ = skill.execute({"template": "tmpl"})  # missing 'data'
        self.assertFalse(success)

    def test_hit_count_increments_on_each_execute(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("fetch", {"url": "http://x.com"}, {"data": 1})
        skill = jit.match("fetch", {"url": "http://x.com"})
        self.assertIsNotNone(skill)
        for _ in range(5):
            skill.execute({"url": "http://x.com"})
        self.assertEqual(skill.hit_count, 5)

    def test_different_arg_keys_produce_different_patterns(self):
        jit = JITCompiler(compile_threshold=2)
        for _ in range(2):
            jit.record_step("write_file", {"path": "a.py"}, "ok1")
        for _ in range(2):
            jit.record_step("write_file", {"path": "b.py", "mode": "w"}, "ok2")
        lib = jit.get_skill_library()
        self.assertEqual(len(lib), 2)

    def test_skill_library_summary_non_empty(self):
        jit = JITCompiler(compile_threshold=1)
        jit.record_step("ping", {"host": "localhost"}, "pong")
        summary = jit.skill_library_summary()
        self.assertIn("ping_compiled", summary)

    def test_total_steps_recorded(self):
        jit = JITCompiler(compile_threshold=5)
        for _ in range(7):
            jit.record_step("op", {"x": 1}, "r")
        self.assertEqual(jit.total_steps_recorded(), 7)

    def test_patterns_pending_before_threshold(self):
        jit = JITCompiler(compile_threshold=5)
        for _ in range(3):
            jit.record_step("partial_op", {"y": 2}, "r")
        pending = jit.patterns_pending_compilation()
        self.assertEqual(len(pending), 1)
        self.assertIn("3/5", pending[0])


if __name__ == "__main__":
    unittest.main()
