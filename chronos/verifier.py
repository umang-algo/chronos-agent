import ast
import inspect
from typing import Callable, List, Dict, Any, Tuple, Optional

class InvariantResult:
    """Result of an invariant verification check."""
    def __init__(self, passed: bool, rule_name: str, message: str, hint: Optional[str] = None):
        self.passed = passed
        self.rule_name = rule_name
        self.message = message
        self.hint = hint or ""

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"<InvariantResult [{status}] rule='{self.rule_name}': {self.message}>"


class InvariantVerifier:
    """
    Engine for checking pre-flight and post-flight invariant assertions
    on tool inputs, environment mutations, and generated code artifacts.
    """
    def __init__(self):
        self.pre_flight_rules: List[Tuple[str, Callable]] = []
        self.post_flight_rules: List[Tuple[str, Callable]] = []
        self._register_default_rules()

    def register_pre_flight_rule(self, name: str, fn: Callable[[Dict[str, Any]], Tuple[bool, str, Optional[str]]]):
        """Registers a pre-flight verification function."""
        self.pre_flight_rules.append((name, fn))

    def register_post_flight_rule(self, name: str, fn: Callable[[Dict[str, Any]], Tuple[bool, str, Optional[str]]]):
        """Registers a post-flight verification function."""
        self.post_flight_rules.append((name, fn))

    def _register_default_rules(self):
        """Registers essential built-in safety and correctness rules."""
        
        # Rule 1: Python Syntax Validity (Pre-flight for file writes)
        def check_python_syntax(payload: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
            filepath = payload.get("filepath", "")
            content = payload.get("content", "")
            if filepath.endswith(".py") and content:
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    return (
                        False, 
                        f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}",
                        f"Python SyntaxError: Fix syntax error on line {e.lineno}: '{e.text.strip() if e.text else ''}' before committing the file."
                    )
            return True, "Syntax check passed.", None

        # Rule 2: Dangerous Shell Command Guard
        def check_dangerous_bash(payload: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
            command = payload.get("command", "").strip()
            dangerous_tokens = ["rm -rf /", "rm -rf ~", ":(){ :|:& };:", "dd if=/dev/zero"]
            for token in dangerous_tokens:
                if token in command:
                    return (
                        False,
                        f"Dangerous shell command detected containing '{token}'.",
                        "Avoid destructive system commands. Use targeted file operations instead."
                    )
            return True, "Bash safety check passed.", None

        self.register_pre_flight_rule("python_syntax_guard", check_python_syntax)
        self.register_pre_flight_rule("dangerous_command_guard", check_dangerous_bash)

    def verify_pre_flight(self, tool_name: str, tool_args: Dict[str, Any]) -> List[InvariantResult]:
        """Runs all applicable pre-flight verification rules."""
        results = []
        payload = {"tool_name": tool_name, **tool_args}
        
        for rule_name, rule_fn in self.pre_flight_rules:
            try:
                passed, msg, hint = rule_fn(payload)
                results.append(InvariantResult(passed, rule_name, msg, hint))
            except Exception as ex:
                results.append(InvariantResult(False, rule_name, f"Rule execution error: {str(ex)}"))
                
        return results

    def verify_post_flight(self, tool_name: str, tool_args: Dict[str, Any], execution_result: Any) -> List[InvariantResult]:
        """Runs all applicable post-flight verification rules."""
        results = []
        payload = {"tool_name": tool_name, "result": execution_result, **tool_args}
        
        for rule_name, rule_fn in self.post_flight_rules:
            try:
                passed, msg, hint = rule_fn(payload)
                results.append(InvariantResult(passed, rule_name, msg, hint))
            except Exception as ex:
                results.append(InvariantResult(False, rule_name, f"Rule execution error: {str(ex)}"))
                
        return results
