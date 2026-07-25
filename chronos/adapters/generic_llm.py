import functools
from typing import Callable, Any, Dict, List, Optional
from chronos.core import ChronosHarness, Checkpoint

class ChronosAgentWrapper:
    """
    Adapter wrapper that decorates any standard agent loop or tool executor
    with Chronos Time-Travel Rollback and Pre-flight Assertion protection.
    """
    def __init__(self, harness: ChronosHarness):
        self.harness = harness

    def wrap_tool(self, tool_name: str, tool_fn: Callable[[Dict[str, Any]], Any]) -> Callable:
        """Wraps a raw tool function with Chronos transactional protection."""
        @functools.wraps(tool_fn)
        def transactional_tool_executor(tool_args: Dict[str, Any], prompt_stack: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Create checkpoint prior to execution
            cp = self.harness.create_checkpoint(prompt_stack=prompt_stack, tool_calls=[{"name": tool_name, "args": tool_args}])
            
            success, result, hint = self.harness.execute_tool_transactional(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_fn=tool_fn,
                prompt_stack=prompt_stack,
                current_checkpoint=cp
            )
            
            return {
                "success": success,
                "result": result,
                "hint": hint,
                "checkpoint": cp
            }

        return transactional_tool_executor
