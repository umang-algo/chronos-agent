import os
import tempfile
import unittest
from pathlib import Path
from chronos.shadow_fs import ShadowFS

class TestShadowFS(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_chronos_fs_")
        self.shadow_fs = ShadowFS(self.temp_dir)
        
        # Create initial test files
        self.file1 = Path(self.temp_dir) / "file1.txt"
        self.file1.write_text("Hello World Initial Content")
        
        self.file2 = Path(self.temp_dir) / "subdir" / "file2.py"
        self.file2.parent.mkdir(parents=True, exist_ok=True)
        self.file2.write_text("def foo():\n    return 42\n")

    def tearDown(self):
        self.shadow_fs.cleanup()

    def test_snapshot_and_rollback_modification(self):
        # 1. Take snapshot
        snapshot = self.shadow_fs.create_snapshot("cp_1")
        
        # 2. Modify file
        self.file1.write_text("Corrupted content created by agent error")
        self.assertEqual(self.file1.read_text(), "Corrupted content created by agent error")
        
        # 3. Rollback
        actions = self.shadow_fs.rollback_to("cp_1")
        
        # 4. Assert original content restored
        self.assertEqual(self.file1.read_text(), "Hello World Initial Content")
        self.assertIn("file1.txt", actions)
        self.assertEqual(actions["file1.txt"], "REVERTED_MODIFIED_FILE")

    def test_snapshot_and_rollback_created_and_deleted_files(self):
        snapshot = self.shadow_fs.create_snapshot("cp_2")
        
        # Delete file2, Create new file3
        self.file2.unlink()
        file3 = Path(self.temp_dir) / "file3.json"
        file3.write_text('{"status": "bad"}')
        
        self.assertFalse(self.file2.exists())
        self.assertTrue(file3.exists())
        
        # Rollback
        actions = self.shadow_fs.rollback_to("cp_2")
        
        # Assert state restored
        self.assertTrue(self.file2.exists())
        self.assertEqual(self.file2.read_text(), "def foo():\n    return 42\n")
        self.assertFalse(file3.exists())

if __name__ == "__main__":
    unittest.main()
