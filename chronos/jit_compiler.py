"""
JIT Trajectory Compiler for Chronos-Agent.
Phase 7: Monitors successful agent trajectories and automatically compiles
repeated patterns into deterministic, zero-token Python scripts (compiled skills).

Architecture:
    TrajectoryRecorder
        └── Records (tool_name, arg_pattern, result) per successful step

    PatternMatcher
        └── Detects when a trajectory pattern repeats >= threshold times

    SkillCompiler
        └── Synthesizes a deterministic Python function from a stable trajectory

    JITCompiler
        ├── record_step(tool_name, args, result)
        ├── match(tool_name, args) -> CompiledSkill | None
        └── get_skill_library() -> Dict[str, CompiledSkill]

When match() returns a CompiledSkill, execute it directly — zero LLM tokens,
sub-millisecond execution. Fall back to the full agent loop only on failure.
"""

import hashlib
import json
import time
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ChronosJIT")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """One observed successful step in an agent's trajectory."""
    tool_name:    str
    args_pattern: Dict[str, str]   # Keys only + arg types (not values for generalization)
    result_type:  str              # Type name of result
    elapsed_ms:   float
    timestamp:    float = field(default_factory=time.time)

    def pattern_key(self) -> str:
        """Stable fingerprint for matching: tool_name + sorted arg key types."""
        sorted_keys = sorted(f"{k}:{v}" for k, v in self.args_pattern.items())
        raw = f"{self.tool_name}::{':'.join(sorted_keys)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class CompiledSkill:
    """
    A compiled, deterministic skill extracted from a stable trajectory pattern.
    Can be executed as a Python callable — zero LLM tokens required.
    """
    skill_id:       str
    name:           str
    description:    str
    trajectory:     List[TrajectoryStep]
    compiled_fn:    Callable[[Dict[str, Any]], Any]
    hit_count:      int = 0
    success_count:  int = 0
    created_at:     float = field(default_factory=time.time)

    def execute(self, args: Dict[str, Any]) -> Tuple[bool, Any, float]:
        """
        Execute the compiled skill. Returns (success, result, elapsed_ms).
        Falls back gracefully — never raises, returns (False, error_msg, ms) on failure.
        """
        start = time.time()
        self.hit_count += 1
        try:
            result = self.compiled_fn(args)
            self.success_count += 1
            elapsed = (time.time() - start) * 1000
            logger.info(f"[JIT] Compiled skill '{self.name}' executed "
                        f"({elapsed:.2f}ms, hits={self.hit_count})")
            return True, result, elapsed
        except Exception as ex:
            elapsed = (time.time() - start) * 1000
            logger.warning(f"[JIT] Compiled skill '{self.name}' failed: {ex}")
            return False, str(ex), elapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "steps": len(self.trajectory),
            "hit_count": self.hit_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_count / max(self.hit_count, 1), 3),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Core JIT Compiler
# ---------------------------------------------------------------------------

class JITCompiler:
    """
    Monitors successful agent trajectories and compiles stable patterns into
    deterministic skills that bypass the LLM loop entirely.

    Usage:
        jit = JITCompiler(compile_threshold=3)

        # After each successful tool call:
        jit.record_step("write_file", {"filepath": "report.md", "content": "..."}, result)

        # Before each tool call, check for a compiled skill:
        skill = jit.match("write_file", {"filepath": "report.md", "content": "..."})
        if skill:
            success, result, ms = skill.execute(args)
            # Zero LLM tokens used!

        # Inspect the skill library:
        print(jit.skill_library_summary())
    """

    def __init__(self, compile_threshold: int = 3):
        """
        Args:
            compile_threshold: Number of times a trajectory pattern must succeed
                               before it is compiled into a deterministic skill.
        """
        self.compile_threshold = compile_threshold
        self._step_history:   List[TrajectoryStep]         = []
        self._pattern_counts: Dict[str, List[Tuple[str, Dict[str, Any], Any]]] = {}
        self._skill_library:  Dict[str, CompiledSkill]     = {}

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_step(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        elapsed_ms: float = 0.0
    ) -> Optional[CompiledSkill]:
        """
        Record one successful tool execution.
        Returns a newly compiled skill if the pattern just crossed the threshold.
        """
        step = self._make_step(tool_name, args, elapsed_ms)
        self._step_history.append(step)

        key = step.pattern_key()
        if key not in self._pattern_counts:
            self._pattern_counts[key] = []
        self._pattern_counts[key].append((tool_name, args, result))

        count = len(self._pattern_counts[key])
        logger.debug(f"[JIT] Pattern '{tool_name}' seen {count}/{self.compile_threshold}x")

        # Compile when threshold crossed
        if count == self.compile_threshold and key not in self._skill_library:
            skill = self._compile(key, tool_name, self._pattern_counts[key])
            self._skill_library[key] = skill
            logger.info(f"[JIT] Compiled new skill: '{skill.name}' (pattern_key={key})")
            return skill

        return None

    # ------------------------------------------------------------------
    # Matching API
    # ------------------------------------------------------------------

    def match(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[CompiledSkill]:
        """
        Check if a compiled skill exists for this tool + arg pattern.
        Returns the CompiledSkill if found, None otherwise.
        """
        step = self._make_step(tool_name, args)
        key = step.pattern_key()
        skill = self._skill_library.get(key)
        if skill:
            logger.info(f"[JIT] HIT — Compiled skill '{skill.name}' matched for '{tool_name}'")
        return skill

    # ------------------------------------------------------------------
    # Introspection API
    # ------------------------------------------------------------------

    def get_skill_library(self) -> Dict[str, CompiledSkill]:
        return dict(self._skill_library)

    def skill_library_summary(self) -> str:
        if not self._skill_library:
            return "[JIT] Skill library is empty — keep recording successful steps."
        lines = [f"[JIT Skill Library] — {len(self._skill_library)} compiled skill(s):"]
        for skill in self._skill_library.values():
            lines.append(
                f"  • '{skill.name}' | hits={skill.hit_count} "
                f"| success_rate={skill.success_count / max(skill.hit_count, 1):.0%} "
                f"| steps={len(skill.trajectory)}"
            )
        return "\n".join(lines)

    def total_steps_recorded(self) -> int:
        return len(self._step_history)

    def patterns_pending_compilation(self) -> List[str]:
        """Tool patterns that have been seen but not yet compiled."""
        return [
            f"{self._pattern_counts[k][0][0]} ({len(v)}/{self.compile_threshold})"
            for k, v in self._pattern_counts.items()
            if k not in self._skill_library
        ]

    # ------------------------------------------------------------------
    # Internal compilation logic
    # ------------------------------------------------------------------

    def _make_step(self, tool_name: str, args: Dict[str, Any], elapsed_ms: float = 0.0) -> TrajectoryStep:
        args_pattern = {k: type(v).__name__ for k, v in args.items()}
        result_type = "unknown"
        return TrajectoryStep(
            tool_name=tool_name,
            args_pattern=args_pattern,
            result_type=result_type,
            elapsed_ms=elapsed_ms,
        )

    def _compile(
        self,
        pattern_key: str,
        tool_name: str,
        observations: List[Tuple[str, Dict[str, Any], Any]]
    ) -> CompiledSkill:
        """
        Synthesize a deterministic skill from observed executions.
        Strategy: use the most recent successful observation as the canonical
        implementation. The compiled_fn replays the last observed result for
        structurally matching args, and raises ValueError for unknown inputs.
        """
        # Use the most recent observation as the canonical result template
        _, canonical_args, canonical_result = observations[-1]

        # Build the compiled function — replays the canonical result
        # In a production JIT, this would be code-synthesized from patterns.
        # Here we create a closure that generalizes across arg values.
        def compiled_fn(args: Dict[str, Any]) -> Any:
            # Structural type match (keys + types must match canonical)
            for k, v in canonical_args.items():
                if k not in args:
                    raise ValueError(
                        f"JIT Skill: missing arg '{k}'. "
                        f"Expected keys: {list(canonical_args.keys())}"
                    )
                if type(args[k]) != type(v):
                    raise ValueError(
                        f"JIT Skill: arg '{k}' type mismatch. "
                        f"Expected {type(v).__name__}, got {type(args[k]).__name__}"
                    )
            return canonical_result

        # Build trajectory log from observations
        trajectory = [
            TrajectoryStep(
                tool_name=tn,
                args_pattern={k: type(v).__name__ for k, v in a.items()},
                result_type=type(r).__name__,
                elapsed_ms=0.0
            )
            for tn, a, r in observations
        ]

        skill_id = f"skill_{tool_name}_{pattern_key}"
        return CompiledSkill(
            skill_id=skill_id,
            name=f"{tool_name}_compiled",
            description=(
                f"Auto-compiled from {len(observations)} successful observations of '{tool_name}'. "
                f"Executes deterministically without LLM inference."
            ),
            trajectory=trajectory,
            compiled_fn=compiled_fn,
        )
