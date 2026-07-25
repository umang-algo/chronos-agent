"""
Agent Persona Freeze & State Fork for Chronos-Agent.
Phase 8: Serializes the complete cognitive state of a running agent
(prompt stack + workspace + custom state + skill memory) to disk as a
portable snapshot. From one frozen snapshot you can:

    - fork(n=3)    → Spin up N isolated 'what-if' harnesses from the same state
    - replay()     → Reconstruct the exact execution state for step-by-step debug
    - diff(other)  → Compare two frozen states to see what changed
    - share()      → Export frozen state for another team member to load

Architecture:
    FrozenPersona          — complete serialized agent state
    PersonaVault           — on-disk store for frozen personas
    PersonaManager         — freeze(), fork(), replay(), diff(), load()
"""

import json
import os
import shutil
import tempfile
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FrozenPersona:
    """
    A complete, portable snapshot of an agent's cognitive + environment state.

    Fields:
        persona_id:      Unique ID for this snapshot.
        label:           Human-readable name (e.g. "before_refactor_attempt").
        prompt_stack:    Full LLM conversation history at freeze time.
        custom_state:    Any agent-specific state dict (memory, counters, etc.).
        workspace_hash:  SHA-256 hash of the workspace content for integrity.
        step_number:     Which execution step this was frozen at.
        ctg_snapshot:    Serialized CausalTraceGraph dict (if available).
        created_at:      Unix timestamp.
        workspace_dir:   Path to a copy of the workspace at freeze time.
    """
    persona_id:     str
    label:          str
    prompt_stack:   List[Dict[str, Any]]
    custom_state:   Dict[str, Any]
    workspace_hash: str
    step_number:    int
    ctg_snapshot:   Optional[Dict[str, Any]]
    created_at:     float
    workspace_dir:  str   # Path to frozen workspace copy (managed by PersonaVault)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "label": self.label,
            "prompt_stack": self.prompt_stack,
            "custom_state": self.custom_state,
            "workspace_hash": self.workspace_hash,
            "step_number": self.step_number,
            "ctg_snapshot": self.ctg_snapshot,
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            f"FrozenPersona '{self.label}' "
            f"(id={self.persona_id[:8]}, step={self.step_number}, "
            f"messages={len(self.prompt_stack)}, "
            f"created={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))})"
        )


@dataclass
class PersonaDiff:
    """Difference between two frozen personas."""
    persona_a_id:  str
    persona_b_id:  str
    step_delta:    int
    added_messages: List[Dict[str, Any]]    # In B but not A
    removed_messages: List[Dict[str, Any]]  # In A but not B
    state_changes: Dict[str, Tuple[Any, Any]]  # key -> (old, new)
    files_added:   List[str]
    files_removed: List[str]
    files_changed: List[str]

    def summary(self) -> str:
        lines = [
            f"Diff: '{self.persona_a_id[:8]}' → '{self.persona_b_id[:8]}'",
            f"  Step delta: {self.step_delta:+d}",
            f"  Messages added: {len(self.added_messages)} | removed: {len(self.removed_messages)}",
            f"  State keys changed: {list(self.state_changes.keys())}",
            f"  Files added: {self.files_added}",
            f"  Files removed: {self.files_removed}",
            f"  Files changed: {self.files_changed}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vault (on-disk storage)
# ---------------------------------------------------------------------------

class PersonaVault:
    """
    Manages on-disk storage of frozen personas.

    Layout:
        vault_dir/
            {persona_id}/
                meta.json        — FrozenPersona metadata
                workspace/       — deep copy of workspace at freeze time
    """

    def __init__(self, vault_dir: str):
        self.vault_dir = os.path.abspath(vault_dir)
        os.makedirs(self.vault_dir, exist_ok=True)

    def save(self, persona: FrozenPersona) -> str:
        """Persist a FrozenPersona to disk. Returns the persona directory path."""
        persona_dir = os.path.join(self.vault_dir, persona.persona_id)
        os.makedirs(persona_dir, exist_ok=True)

        # Save metadata (everything except workspace_dir which is local path)
        meta_path = os.path.join(persona_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(persona.to_dict(), f, indent=2)

        # Copy workspace into vault
        vault_ws = os.path.join(persona_dir, "workspace")
        if os.path.isdir(persona.workspace_dir) and persona.workspace_dir != vault_ws:
            if os.path.isdir(vault_ws):
                shutil.rmtree(vault_ws)
            shutil.copytree(
                persona.workspace_dir,
                vault_ws,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".chronos_*")
            )

        return persona_dir

    def load(self, persona_id: str) -> FrozenPersona:
        """Load a FrozenPersona from disk by ID."""
        persona_dir = os.path.join(self.vault_dir, persona_id)
        meta_path = os.path.join(persona_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Persona '{persona_id}' not found in vault.")

        with open(meta_path) as f:
            data = json.load(f)

        return FrozenPersona(
            persona_id=data["persona_id"],
            label=data["label"],
            prompt_stack=data["prompt_stack"],
            custom_state=data["custom_state"],
            workspace_hash=data["workspace_hash"],
            step_number=data["step_number"],
            ctg_snapshot=data.get("ctg_snapshot"),
            created_at=data["created_at"],
            workspace_dir=os.path.join(persona_dir, "workspace"),
        )

    def list_personas(self) -> List[Dict[str, Any]]:
        """List all saved personas in this vault."""
        entries = []
        for name in os.listdir(self.vault_dir):
            meta_path = os.path.join(self.vault_dir, name, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    data = json.load(f)
                entries.append({
                    "persona_id": data["persona_id"],
                    "label": data["label"],
                    "step_number": data["step_number"],
                    "created_at": data["created_at"],
                })
        return sorted(entries, key=lambda x: x["created_at"])

    def delete(self, persona_id: str) -> None:
        persona_dir = os.path.join(self.vault_dir, persona_id)
        if os.path.isdir(persona_dir):
            shutil.rmtree(persona_dir)


# ---------------------------------------------------------------------------
# PersonaManager — main API
# ---------------------------------------------------------------------------

class PersonaManager:
    """
    High-level API for freezing, forking, replaying, and diffing agent states.

    Usage:
        vault = PersonaVault("./.chronos_vault")
        manager = PersonaManager(vault)

        # Freeze current agent state
        frozen = manager.freeze(
            label="before_tax_refactor",
            harness=harness,
            prompt_stack=messages,
            custom_state={"memory": agent_memory}
        )

        # Fork into 3 isolated harnesses for what-if simulation
        forks = manager.fork(frozen, n=3)
        # forks[0], forks[1], forks[2] are independent ChronosHarness instances

        # Diff two frozen states to see what changed
        diff = manager.diff(frozen_a, frozen_b)
        print(diff.summary())

        # Load a previously saved persona for replay
        loaded = manager.load("abc12345")
    """

    def __init__(self, vault: PersonaVault):
        self.vault = vault

    def freeze(
        self,
        label: str,
        workspace_dir: str,
        prompt_stack: List[Dict[str, Any]],
        step_number: int = 0,
        custom_state: Optional[Dict[str, Any]] = None,
        ctg_snapshot: Optional[Dict[str, Any]] = None,
    ) -> FrozenPersona:
        """
        Freeze the complete agent state to disk.
        Returns a FrozenPersona that can be forked, replayed, or diffed.
        """
        persona_id = self._generate_id(label, step_number)
        ws_hash = self._hash_workspace(workspace_dir)

        # Create a temp copy of the workspace for this persona
        temp_ws = tempfile.mkdtemp(prefix=f"chronos_persona_{persona_id[:8]}_")
        if os.path.isdir(workspace_dir):
            shutil.copytree(
                workspace_dir,
                os.path.join(temp_ws, "ws"),
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".chronos_*")
            )

        persona = FrozenPersona(
            persona_id=persona_id,
            label=label,
            prompt_stack=[dict(m) for m in prompt_stack],
            custom_state=dict(custom_state or {}),
            workspace_hash=ws_hash,
            step_number=step_number,
            ctg_snapshot=ctg_snapshot,
            created_at=time.time(),
            workspace_dir=os.path.join(temp_ws, "ws"),
        )

        # Save to vault — vault copies workspace into its own managed directory
        self.vault.save(persona)

        # P0 Fix #2: Delete the intermediate temp dir now that the vault has
        # its own copy. Prevents leaked GBs on long-running agent sessions.
        try:
            shutil.rmtree(temp_ws, ignore_errors=True)
        except Exception:
            pass

        # Return persona with workspace_dir pointing to the vault copy
        vault_ws = os.path.join(self.vault.vault_dir, persona_id, "workspace")
        persona.workspace_dir = vault_ws
        return persona

    def fork(self, persona: FrozenPersona, n: int = 2) -> List[Dict[str, Any]]:
        """
        Fork a frozen persona into N independent isolated agent contexts.
        Returns a list of fork descriptors: each contains a workspace_dir
        and the full prompt_stack/custom_state ready to boot a new harness.

        Usage:
            forks = manager.fork(frozen, n=3)
            for fork in forks:
                harness = ChronosHarness(fork["workspace_dir"])
                # Each harness is completely isolated — run different strategies
        """
        forks = []
        for i in range(n):
            fork_dir = tempfile.mkdtemp(prefix=f"chronos_fork_{i}_{persona.persona_id[:8]}_")
            fork_ws = os.path.join(fork_dir, "workspace")

            if os.path.isdir(persona.workspace_dir):
                shutil.copytree(
                    persona.workspace_dir,
                    fork_ws,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".chronos_*")
                )
            else:
                os.makedirs(fork_ws, exist_ok=True)

            forks.append({
                "fork_index": i,
                "fork_id": f"{persona.persona_id[:8]}_fork_{i}",
                "workspace_dir": fork_ws,
                "prompt_stack": [dict(m) for m in persona.prompt_stack],
                "custom_state": dict(persona.custom_state),
                "step_number": persona.step_number,
                "source_persona_id": persona.persona_id,
                "_temp_root": fork_dir,  # Keep for cleanup
            })

        return forks

    def cleanup_fork(self, fork: Dict[str, Any]) -> None:
        """
        P0 Fix #2: Explicitly clean up a fork's isolated temp directory.
        Call this after you're done using a fork to prevent disk leaks.

        Usage:
            forks = manager.fork(frozen, n=3)
            try:
                # ... use forks ...
            finally:
                for fork in forks:
                    manager.cleanup_fork(fork)
        """
        temp_root = fork.get("_temp_root")
        if temp_root and os.path.isdir(temp_root):
            shutil.rmtree(temp_root, ignore_errors=True)

    def cleanup_all_forks(self, forks: List[Dict[str, Any]]) -> None:
        """Convenience: clean up all forks in a list."""
        for fork in forks:
            self.cleanup_fork(fork)

    def replay(self, persona: FrozenPersona) -> Dict[str, Any]:
        """
        Reconstruct the exact state for step-by-step debugging.
        Returns a replay descriptor ready to boot a ChronosHarness.
        """
        replay_dir = tempfile.mkdtemp(prefix=f"chronos_replay_{persona.persona_id[:8]}_")
        replay_ws = os.path.join(replay_dir, "workspace")

        if os.path.isdir(persona.workspace_dir):
            shutil.copytree(
                persona.workspace_dir,
                replay_ws,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".chronos_*")
            )
        else:
            os.makedirs(replay_ws, exist_ok=True)

        return {
            "mode": "replay",
            "persona_id": persona.persona_id,
            "label": persona.label,
            "workspace_dir": replay_ws,
            "prompt_stack": [dict(m) for m in persona.prompt_stack],
            "custom_state": dict(persona.custom_state),
            "step_number": persona.step_number,
            "ctg_snapshot": persona.ctg_snapshot,
            "_temp_root": replay_dir,
        }

    def diff(self, persona_a: FrozenPersona, persona_b: FrozenPersona) -> PersonaDiff:
        """
        Compute the semantic diff between two frozen personas.
        Shows exactly what changed in prompts, state, and workspace files.
        """
        # Message diff
        msgs_a = {json.dumps(m, sort_keys=True) for m in persona_a.prompt_stack}
        msgs_b = {json.dumps(m, sort_keys=True) for m in persona_b.prompt_stack}
        added_msgs   = [json.loads(m) for m in (msgs_b - msgs_a)]
        removed_msgs = [json.loads(m) for m in (msgs_a - msgs_b)]

        # State diff
        state_changes: Dict[str, Tuple[Any, Any]] = {}
        all_keys = set(persona_a.custom_state) | set(persona_b.custom_state)
        for k in all_keys:
            va = persona_a.custom_state.get(k, "__MISSING__")
            vb = persona_b.custom_state.get(k, "__MISSING__")
            if va != vb:
                state_changes[k] = (va, vb)

        # File diff
        files_a = self._list_files(persona_a.workspace_dir)
        files_b = self._list_files(persona_b.workspace_dir)

        added   = sorted(set(files_b) - set(files_a))
        removed = sorted(set(files_a) - set(files_b))
        common  = set(files_a) & set(files_b)
        changed = []
        for rel_path in sorted(common):
            path_a = os.path.join(persona_a.workspace_dir, rel_path)
            path_b = os.path.join(persona_b.workspace_dir, rel_path)
            if self._file_hash(path_a) != self._file_hash(path_b):
                changed.append(rel_path)

        return PersonaDiff(
            persona_a_id=persona_a.persona_id,
            persona_b_id=persona_b.persona_id,
            step_delta=persona_b.step_number - persona_a.step_number,
            added_messages=added_msgs,
            removed_messages=removed_msgs,
            state_changes=state_changes,
            files_added=added,
            files_removed=removed,
            files_changed=changed,
        )

    def load(self, persona_id: str) -> FrozenPersona:
        """Load a previously saved FrozenPersona from the vault."""
        return self.vault.load(persona_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_id(self, label: str, step: int) -> str:
        raw = f"{label}:{step}:{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _hash_workspace(self, workspace_dir: str) -> str:
        if not os.path.isdir(workspace_dir):
            return "empty"
        h = hashlib.sha256()
        for root, dirs, files in sorted(os.walk(workspace_dir)):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for fname in sorted(files):
                if fname.endswith(".pyc"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        h.update(f.read())
                except Exception:
                    pass
        return h.hexdigest()[:16]

    def _list_files(self, workspace_dir: str) -> List[str]:
        if not os.path.isdir(workspace_dir):
            return []
        result = []
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for fname in files:
                if not fname.endswith(".pyc"):
                    rel = os.path.relpath(os.path.join(root, fname), workspace_dir)
                    result.append(rel)
        return result

    def _file_hash(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
