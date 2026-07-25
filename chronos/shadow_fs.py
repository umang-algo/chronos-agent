import os
import shutil
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

class FileState:
    """Represents the cryptographic state of a single file."""
    def __init__(self, rel_path: str, abs_path: Path):
        self.rel_path = rel_path
        self.abs_path = abs_path
        self.exists = abs_path.exists()
        self.is_dir = abs_path.is_dir() if self.exists else False
        self.content_hash: Optional[str] = self._compute_hash() if (self.exists and not self.is_dir) else None
        self.mtime: Optional[float] = abs_path.stat().st_mtime if self.exists else None

    def _compute_hash(self) -> str:
        try:
            hasher = hashlib.sha256()
            with open(self.abs_path, 'rb') as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def __repr__(self):
        return f"<FileState rel='{self.rel_path}' exists={self.exists} hash={self.content_hash[:8] if self.content_hash else 'N/A'}>"


class ShadowSnapshot:
    """A point-in-time snapshot of the tracked directory."""
    def __init__(self, snapshot_id: str, backup_dir: Path, file_states: Dict[str, FileState]):
        self.snapshot_id = snapshot_id
        self.backup_dir = backup_dir
        self.file_states = file_states
        self.created_at = time.time()

    def get_diff(self, current_states: Dict[str, FileState]) -> Dict[str, str]:
        """Compares this snapshot with current state and categorizes changes."""
        diffs = {}
        all_paths = set(self.file_states.keys()) | set(current_states.keys())
        
        for path in all_paths:
            old_st = self.file_states.get(path)
            new_st = current_states.get(path)
            
            if old_st and not new_st:
                diffs[path] = "DELETED"
            elif not old_st and new_st:
                diffs[path] = "CREATED"
            elif old_st and new_st and old_st.content_hash != new_st.content_hash:
                diffs[path] = "MODIFIED"
                
        return diffs


class ShadowFS:
    """
    Sub-second Transactional Shadow Filesystem Overlay.
    Tracks file tree state and allows instant rollback to any historical snapshot.
    """
    def __init__(self, target_dir: str, ignore_patterns: Optional[List[str]] = None, max_snapshots: int = 50):
        self.target_dir = Path(target_dir).resolve()
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True, exist_ok=True)
            
        self.ignore_patterns = set(ignore_patterns or [
            ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".DS_Store"
        ])
        
        self.max_snapshots = max_snapshots
        self.storage_dir = Path(tempfile.mkdtemp(prefix="chronos_shadow_"))
        self.snapshots: Dict[str, ShadowSnapshot] = {}

    def _should_ignore(self, path: Path) -> bool:
        for part in path.parts:
            if part in self.ignore_patterns or part.startswith(".chronos"):
                return True
        return False

    def scan_state(self) -> Dict[str, FileState]:
        """Scans the target directory and returns a map of relative paths to FileStates."""
        states = {}
        for root, dirs, files in os.walk(self.target_dir):
            root_path = Path(root)
            
            # Filter directories in-place to avoid traversing ignored folders
            dirs[:] = [d for d in dirs if not self._should_ignore(root_path / d)]
            
            for file_name in files:
                abs_path = root_path / file_name
                if self._should_ignore(abs_path):
                    continue
                rel_path = str(abs_path.relative_to(self.target_dir))
                states[rel_path] = FileState(rel_path, abs_path)
                
        return states

    def create_snapshot(self, snapshot_id: str) -> ShadowSnapshot:
        """Takes a fast snapshot of the tracked directory state."""
        current_states = self.scan_state()
        snapshot_backup_path = self.storage_dir / snapshot_id
        snapshot_backup_path.mkdir(parents=True, exist_ok=True)

        # Copy current file contents into shadow storage
        for rel_path, state in current_states.items():
            if state.exists and not state.is_dir:
                backup_file = snapshot_backup_path / rel_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(state.abs_path, backup_file)

        snapshot = ShadowSnapshot(snapshot_id, snapshot_backup_path, current_states)
        self.snapshots[snapshot_id] = snapshot

        # P0 Fix #3: Evict oldest snapshot if limit exceeded to prevent unbounded disk growth
        if len(self.snapshots) > self.max_snapshots:
            oldest_id = next(iter(self.snapshots))
            self.delete_snapshot(oldest_id)

        return snapshot

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Deletes a snapshot and frees its backup files from disk."""
        snapshot = self.snapshots.pop(snapshot_id, None)
        if snapshot and snapshot.backup_dir.exists():
            shutil.rmtree(snapshot.backup_dir, ignore_errors=True)

    def rollback_to(self, snapshot_id: str) -> Dict[str, str]:
        """
        Sub-second rollback to a previous snapshot state.
        Restores modified/deleted files and removes newly created files.
        Returns a dictionary of actions taken during rollback.
        """
        if snapshot_id not in self.snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found in Chronos ShadowFS.")

        target_snapshot = self.snapshots[snapshot_id]
        current_states = self.scan_state()
        actions_taken = {}

        # 1. Identify files created since the snapshot and delete them
        for rel_path, curr_state in current_states.items():
            if rel_path not in target_snapshot.file_states:
                if curr_state.abs_path.exists():
                    if curr_state.is_dir:
                        shutil.rmtree(curr_state.abs_path)
                    else:
                        curr_state.abs_path.unlink()
                    actions_taken[rel_path] = "REMOVED_CREATED_FILE"

        # 2. Restore modified or deleted files from backup
        for rel_path, old_state in target_snapshot.file_states.items():
            curr_state = current_states.get(rel_path)
            target_file_path = self.target_dir / rel_path
            backup_file_path = target_snapshot.backup_dir / rel_path

            if not curr_state:
                # File was deleted since snapshot, restore it
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                if backup_file_path.exists():
                    shutil.copy2(backup_file_path, target_file_path)
                    actions_taken[rel_path] = "RESTORED_DELETED_FILE"
            elif curr_state.content_hash != old_state.content_hash:
                # File was modified, revert to original backup
                if backup_file_path.exists():
                    shutil.copy2(backup_file_path, target_file_path)
                    actions_taken[rel_path] = "REVERTED_MODIFIED_FILE"

        # 3. Clean up empty directories created since the snapshot
        for root, dirs, files in os.walk(self.target_dir, topdown=False):
            root_path = Path(root)
            if root_path == self.target_dir or self._should_ignore(root_path):
                continue
            rel_dir = str(root_path.relative_to(self.target_dir))
            # If directory has no files and was not present as a directory in target_snapshot
            if not os.listdir(root_path):
                old_dir_exists = any(p.startswith(rel_dir + os.sep) or p == rel_dir for p in target_snapshot.file_states.keys())
                if not old_dir_exists:
                    shutil.rmtree(root_path, ignore_errors=True)
                    actions_taken[rel_dir] = "REMOVED_CREATED_DIR"

        return actions_taken

    def cleanup(self):
        """Cleans up temporary shadow storage on disk."""
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir, ignore_errors=True)

    def __del__(self):
        self.cleanup()
