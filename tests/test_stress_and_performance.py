import os
import time
import tempfile
import unittest
from pathlib import Path
from chronos.shadow_fs import ShadowFS

class TestStressAndPerformance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="stress_chronos_")
        self.shadow_fs = ShadowFS(self.temp_dir)

    def tearDown(self):
        self.shadow_fs.cleanup()

    def test_performance_sub_second_guarantee(self):
        """Benchmark snapshot creation and rollback on a project with 50+ files."""
        # Generate 50 code files in nested directories
        for i in range(50):
            sub_folder = Path(self.temp_dir) / f"module_{i % 5}"
            sub_folder.mkdir(parents=True, exist_ok=True)
            file_path = sub_folder / f"service_{i}.py"
            file_path.write_text(f"# Service {i} Implementation\ndef run_{i}(): return {i} * 42\n")

        # Measure Snapshot Speed
        start_time = time.time()
        snapshot = self.shadow_fs.create_snapshot("perf_snapshot_1")
        snapshot_duration_ms = (time.time() - start_time) * 1000

        print(f"\n[BENCHMARK] 50 Files Snapshot Time: {snapshot_duration_ms:.2f} ms")
        self.assertLess(snapshot_duration_ms, 150.0, "Snapshot creation exceeded 150ms target.")

        # Mutate 20 files and delete 10 files
        for i in range(20):
            file_path = Path(self.temp_dir) / f"module_{i % 5}" / f"service_{i}.py"
            file_path.write_text("# CORRUPTED CONTENT")

        for i in range(20, 30):
            file_path = Path(self.temp_dir) / f"module_{i % 5}" / f"service_{i}.py"
            if file_path.exists():
                file_path.unlink()

        # Measure Rollback Speed
        start_time = time.time()
        actions = self.shadow_fs.rollback_to("perf_snapshot_1")
        rollback_duration_ms = (time.time() - start_time) * 1000

        print(f"[BENCHMARK] 50 Files Rollback Time: {rollback_duration_ms:.2f} ms")
        self.assertLess(rollback_duration_ms, 100.0, "Rollback duration exceeded 100ms target.")

        # Verify 100% data integrity restored
        for i in range(50):
            file_path = Path(self.temp_dir) / f"module_{i % 5}" / f"service_{i}.py"
            self.assertTrue(file_path.exists(), f"File service_{i}.py failed to restore.")
            self.assertEqual(file_path.read_text(), f"# Service {i} Implementation\ndef run_{i}(): return {i} * 42\n")

    def test_complex_nested_directory_mutations(self):
        """Test snapshot & rollback with nested subdirectories and binary assets."""
        nested_dir = Path(self.temp_dir) / "src" / "deep" / "nested" / "pkg"
        nested_dir.mkdir(parents=True, exist_ok=True)
        code_file = nested_dir / "index.ts"
        code_file.write_text("export const version = '1.0.0';")

        binary_file = Path(self.temp_dir) / "assets" / "logo.bin"
        binary_file.parent.mkdir(parents=True, exist_ok=True)
        binary_content = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF])
        binary_file.write_bytes(binary_content)

        snapshot = self.shadow_fs.create_snapshot("nested_snap")

        # Destructive Agent Actions
        code_file.write_text("BROKEN TS CODE")
        binary_file.write_bytes(bytes([0x00, 0x00]))
        new_dir = Path(self.temp_dir) / "temp_junk"
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "junk.txt").write_text("Junk data")

        # Rollback
        self.shadow_fs.rollback_to("nested_snap")

        # Assert full recovery
        self.assertEqual(code_file.read_text(), "export const version = '1.0.0';")
        self.assertEqual(binary_file.read_bytes(), binary_content)
        self.assertFalse(new_dir.exists())

if __name__ == "__main__":
    unittest.main()
