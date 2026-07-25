"""
Chronos-Agent: The Transactional Intelligence Layer for AI Agents.

Brings transactional state snapshotting, causal failure tracing,
pre-flight assertion verification, scope-locked policy enforcement,
speculative parallel branching, JIT trajectory compilation, and
sub-second time-travel rollback to any AI agent loop.
"""

from chronos.core import ChronosHarness, Checkpoint, RollbackException
from chronos.shadow_fs import ShadowFS
from chronos.dual_ledger import DualLedger, ExecutionTurn
from chronos.verifier import InvariantVerifier
from chronos.causal_graph import CausalTraceGraph, CausalNode, CausalEdge, NodeKind, NodeStatus
from chronos.policy import PolicyEngine, Policy, PolicyViolation
from chronos.speculative import SpeculativeExecutor, BranchStrategy, BranchOutcome, SpeculativeResult
from chronos.jit_compiler import JITCompiler, CompiledSkill, TrajectoryStep
from chronos.persona import PersonaManager, PersonaVault, FrozenPersona, PersonaDiff
from chronos.adapters.generic_llm import ChronosAgentWrapper
from chronos.adapters.langgraph import ChronosLangGraphMiddleware

__version__ = "0.2.0"
__all__ = [
    # Core
    "ChronosHarness",
    "Checkpoint",
    "RollbackException",
    "ShadowFS",
    "DualLedger",
    "ExecutionTurn",
    "InvariantVerifier",
    # Causal Trace Graph (Phase 3)
    "CausalTraceGraph",
    "CausalNode",
    "CausalEdge",
    "NodeKind",
    "NodeStatus",
    # Policy Engine (Phase 5)
    "PolicyEngine",
    "Policy",
    "PolicyViolation",
    # Speculative Branching (Phase 6)
    "SpeculativeExecutor",
    "BranchStrategy",
    "BranchOutcome",
    "SpeculativeResult",
    # JIT Compiler (Phase 7)
    "JITCompiler",
    "CompiledSkill",
    "TrajectoryStep",
    # Persona Freeze & Fork (Phase 8)
    "PersonaManager",
    "PersonaVault",
    "FrozenPersona",
    "PersonaDiff",
    # Adapters
    "ChronosAgentWrapper",
    "ChronosLangGraphMiddleware",
]
