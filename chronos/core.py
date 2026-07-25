import threading
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple

from chronos.shadow_fs import ShadowFS
from chronos.dual_ledger import DualLedger, Checkpoint
from chronos.verifier import InvariantVerifier, InvariantResult
from chronos.causal_graph import CausalTraceGraph, CausalNode, NodeKind, NodeStatus

logger = logging.getLogger("ChronosHarness")

class RollbackException(Exception):
    """Raised when an execution turn fails and triggers a time-travel rollback."""
    def __init__(self, target_checkpoint_id: str, reason: str, counterfactual_hint: str):
        super().__init__(reason)
        self.target_checkpoint_id = target_checkpoint_id
        self.reason = reason
        self.counterfactual_hint = counterfactual_hint


class ChronosHarness:
    """
    Chronos-Agent: The Transactional Time-Travel Rollback Harness.

    Provides sub-second snapshotting, pre-flight assertion verification,
    and automatic time-travel state rollback with counterfactual hint injection.

    Thread-safe: all checkpoint creation and tool execution is protected by
    an internal RLock, allowing safe use inside async/multi-threaded agent loops.
    """
    def __init__(
        self,
        workspace_dir: str,
        max_rollback_attempts: int = 3,
        ignore_patterns: Optional[List[str]] = None
    ):
        self.workspace_dir = workspace_dir
        self.shadow_fs = ShadowFS(workspace_dir, ignore_patterns=ignore_patterns)
        self.ledger = DualLedger()
        self.verifier = InvariantVerifier()
        self.causal_graph = CausalTraceGraph()   # <-- Live Causal Execution Trace Graph
        self.max_rollback_attempts = max_rollback_attempts
        self.rollback_counts: Dict[str, int] = {}
        self.current_step = 0
        self._last_ctg_node: Optional[CausalNode] = None  # Tracks causal parent for next node
        # P0 Fix #1 — Thread safety: RLock allows the same thread to re-enter
        # (e.g. rollback() called from within execute_tool_transactional())
        self._lock = threading.RLock()

    def create_checkpoint(
        self,
        prompt_stack: List[Dict[str, Any]],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        custom_state: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Creates an atomic checkpoint across both Ledgers:
        1. Takes sub-second filesystem snapshot.
        2. Records conversation prompt state & metadata in DualLedger.

        Thread-safe — protected by internal RLock.
        """
        with self._lock:
            self.current_step += 1
            checkpoint_id = f"cp_step_{self.current_step}_{int(time.time() * 1000)}"

            # 1. Take shadow filesystem snapshot
            fs_snapshot = self.shadow_fs.create_snapshot(checkpoint_id)

            # 2. Record dual ledger entry
            cp = self.ledger.record_checkpoint(
                checkpoint_id=checkpoint_id,
                step_number=self.current_step,
                prompt_stack=prompt_stack,
                fs_snapshot_id=fs_snapshot.snapshot_id,
                tool_calls=tool_calls,
                custom_state=custom_state
            )

            # 3. Record Checkpoint in Causal Trace Graph
            ctg_node = self.causal_graph.record_checkpoint(
                checkpoint_id=checkpoint_id,
                step=self.current_step,
                caused_by=self._last_ctg_node
            )
            self._last_ctg_node = ctg_node

            logger.info(f"[Chronos] Created Atomic Checkpoint '{checkpoint_id}' at Step {self.current_step}")
            return cp

    def execute_tool_transactional(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_fn: Callable[[Dict[str, Any]], Any],
        prompt_stack: List[Dict[str, Any]],
        current_checkpoint: Checkpoint
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Executes a tool within a transactional Chronos boundary:
        1. Runs Pre-flight Invariant Verification.
        2. If Pre-flight fails: cancels execution, triggers rollback to current_checkpoint.
        3. Runs tool_fn.
        4. Runs Post-flight Invariant Verification.
        5. If Post-flight fails: triggers rollback to current_checkpoint.

        Thread-safe — protected by internal RLock. The RLock is re-entrant so
        rollback() (which also acquires the lock) can be called internally.
        """
        with self._lock:
            # Record tool call intent in CTG
            tool_node = self.causal_graph.record_tool_call(
                tool_name=tool_name,
                args=tool_args,
                step=self.current_step,
                caused_by=self._last_ctg_node
            )
            self._last_ctg_node = tool_node

            # Step 1: Pre-flight Verification
            pre_results = self.verifier.verify_pre_flight(tool_name, tool_args)
            failed_pre = [r for r in pre_results if not r.passed]

            if failed_pre:
                failure_msg = f"Pre-flight assertion failed: {failed_pre[0].message}"
                hint = failed_pre[0].hint or failure_msg
                logger.warning(f"[Chronos] Pre-flight Failure in '{tool_name}': {failure_msg}")

                # Record invariant failure in CTG
                inv_node = self.causal_graph.record_invariant_check(
                    rule_name=failed_pre[0].rule_name, passed=False,
                    step=self.current_step, caused_by=tool_node,
                    error=failure_msg, hint=hint
                )
                fail_node = self.causal_graph.record_failure(
                    reason=failure_msg, step=self.current_step,
                    caused_by=inv_node, hint=hint
                )
                self._last_ctg_node = fail_node
                self.rollback(current_checkpoint.checkpoint_id, reason=failure_msg, counterfactual_hint=hint)
                return False, None, hint

            # Record pre-flight passed
            self.causal_graph.record_invariant_check(
                rule_name="pre_flight", passed=True,
                step=self.current_step, caused_by=tool_node
            )

            # Step 2: Tool Execution
            try:
                tool_result = tool_fn(tool_args)
                tool_node.mark_success(result_summary=str(tool_result)[:100] if tool_result else None)
            except Exception as ex:
                failure_msg = f"Tool execution exception in '{tool_name}': {str(ex)}"
                hint = f"Tool '{tool_name}' crashed with error: {str(ex)}. Check parameters and environment before retrying."
                logger.error(f"[Chronos] Execution Exception: {failure_msg}")

                fail_node = self.causal_graph.record_failure(
                    reason=failure_msg, step=self.current_step,
                    caused_by=tool_node, hint=hint
                )
                self._last_ctg_node = fail_node
                self.rollback(current_checkpoint.checkpoint_id, reason=failure_msg, counterfactual_hint=hint)
                return False, None, hint

            # Step 3: Post-flight Verification
            post_results = self.verifier.verify_post_flight(tool_name, tool_args, tool_result)
            failed_post = [r for r in post_results if not r.passed]

            if failed_post:
                failure_msg = f"Post-flight assertion failed: {failed_post[0].message}"
                hint = failed_post[0].hint or failure_msg
                logger.warning(f"[Chronos] Post-flight Failure in '{tool_name}': {failure_msg}")

                inv_node = self.causal_graph.record_invariant_check(
                    rule_name=failed_post[0].rule_name, passed=False,
                    step=self.current_step, caused_by=tool_node,
                    error=failure_msg, hint=hint
                )
                fail_node = self.causal_graph.record_failure(
                    reason=failure_msg, step=self.current_step,
                    caused_by=inv_node, hint=hint
                )
                self._last_ctg_node = fail_node
                self.rollback(current_checkpoint.checkpoint_id, reason=failure_msg, counterfactual_hint=hint)
                return False, tool_result, hint

            return True, tool_result, None

    def rollback(
        self,
        target_checkpoint_id: str,
        reason: str,
        counterfactual_hint: str
    ) -> List[Dict[str, Any]]:
        """
        Performs sub-second time-travel rollback:
        1. Reverts filesystem to target_checkpoint_id snapshot.
        2. Truncates prompt stack to checkpoint step.
        3. Appends counterfactual diagnostic hint to prompt stack.

        Thread-safe — protected by internal RLock (re-entrant, safe to call
        from within execute_tool_transactional).

        P0 Fix #5: rollback() is idempotent within the same causal chain.
        Calling rollback() manually after execute_tool_transactional() already
        performed it will increment the attempt counter but not corrupt state,
        because ShadowFS rollback to the same checkpoint is deterministic.
        The caller should use the returned prompt_stack, not call rollback() again.
        """
        with self._lock:
            cp = self.ledger.get_checkpoint(target_checkpoint_id)
            if not cp:
                raise KeyError(f"Target checkpoint '{target_checkpoint_id}' not found.")

            # Check rollback attempt count to prevent infinite loops
            attempts = self.rollback_counts.get(target_checkpoint_id, 0) + 1
            self.rollback_counts[target_checkpoint_id] = attempts

            if attempts > self.max_rollback_attempts:
                raise RuntimeError(
                    f"Max rollback attempts ({self.max_rollback_attempts}) exceeded "
                    f"for checkpoint '{target_checkpoint_id}'. Reason: {reason}"
                )

            logger.info(
                f"[Chronos TIME-TRAVEL ROLLBACK] Reverting to Checkpoint "
                f"'{target_checkpoint_id}' (Step {cp.step_number}). "
                f"Attempt {attempts}/{self.max_rollback_attempts}"
            )

            # 1. Restore Filesystem State
            actions = self.shadow_fs.rollback_to(cp.fs_snapshot_id)
            logger.info(f"[Chronos ShadowFS] Rollback actions: {actions}")

            # 2. Truncate Dual Ledger
            self.ledger.truncate_to(target_checkpoint_id)

            # 3. Record Rollback + Counterfactual in Causal Trace Graph
            rb_node = self.causal_graph.record_rollback(
                to_checkpoint_id=target_checkpoint_id,
                step=self.current_step,
                caused_by=self._last_ctg_node
            )
            cf_node = self.causal_graph.record_counterfactual(
                hint=counterfactual_hint,
                step=self.current_step,
                caused_by=rb_node
            )
            self._last_ctg_node = cf_node

            # 4. Prepare Restored Prompt Stack with Counterfactual System Hint
            restored_prompts = [dict(msg) for msg in cp.turn.prompt_snapshot]

            # Enrich counterfactual with live CTG root cause analysis
            failures = self.causal_graph.find_all_failures()
            root_cause_explanation = ""
            if failures:
                root_cause_explanation = "\n[CTG Root Cause]:\n" + self.causal_graph.explain_failure(failures[-1].node_id)

            counterfactual_system_msg = {
                "role": "system",
                "content": (
                    f"⚠️ [CHRONOS TIME-TRAVEL ROLLBACK TRIGGERED AT STEP {cp.step_number}]\n"
                    f"Reason: {reason}\n"
                    f"Counterfactual Instruction: {counterfactual_hint}\n"
                    f"{root_cause_explanation}\n"
                    f"The environment has been cleanly reverted to Step {cp.step_number} state. "
                    f"Do NOT repeat the failed action. Choose an alternate clean strategy."
                )
            }
            restored_prompts.append(counterfactual_system_msg)

            return restored_prompts

    def cleanup(self):
        """Clean up temporary resources."""
        self.shadow_fs.cleanup()
