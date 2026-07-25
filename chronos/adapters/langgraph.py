from typing import Callable, Dict, Any, List
from chronos.core import ChronosHarness

class ChronosLangGraphMiddleware:
    """
    Middleware adapter for integrating Chronos Time-Travel Rollback with LangGraph state nodes.
    """
    def __init__(self, harness: ChronosHarness):
        self.harness = harness

    def wrap_node(self, node_name: str, node_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Wraps a LangGraph node function with transactional Chronos rollback protection."""
        def transactional_node_executor(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            # Format prompt stack if messages are objects or dicts
            prompt_stack = []
            for msg in messages:
                if isinstance(msg, dict):
                    prompt_stack.append(msg)
                elif hasattr(msg, "content"):
                    prompt_stack.append({"role": getattr(msg, "type", "user"), "content": msg.content})

            # Create Checkpoint
            cp = self.harness.create_checkpoint(prompt_stack=prompt_stack)

            try:
                # Execute node
                new_state = node_fn(state)
                return new_state
            except Exception as ex:
                failure_reason = f"LangGraph node '{node_name}' crashed: {str(ex)}"
                counterfactual_hint = f"Node '{node_name}' failed with error: {str(ex)}. Choose an alternate branch."
                
                # Perform Rollback
                restored_prompts = self.harness.rollback(
                    target_checkpoint_id=cp.checkpoint_id,
                    reason=failure_reason,
                    counterfactual_hint=counterfactual_hint
                )
                
                # Return state with counterfactual hint injected into messages
                state_copy = dict(state)
                state_copy["messages"] = restored_prompts
                state_copy["chronos_rollback_occurred"] = True
                state_copy["chronos_hint"] = counterfactual_hint
                return state_copy

        return transactional_node_executor
