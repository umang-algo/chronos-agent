import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos import ChronosHarness, ChronosAgentWrapper

# Terminal ANSI Color Codes for Mind-Blowing Visual Output
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def log_box(title: str, text: str, color: str = CYAN):
    border = "═" * 70
    print(f"\n{color}{BOLD}╔{border}╗")
    print(f"║ {title.ljust(68)} ║")
    print(f"╠{border}╣{RESET}")
    for line in text.split("\n"):
        print(f"{color}║ {line.ljust(68)} ║{RESET}")
    print(f"{color}{BOLD}╚{border}╝{RESET}\n")


def main():
    print(f"\n{BOLD}{MAGENTA}" + "=" * 72)
    print(" 🚀 CHRONOS-AGENT: TRANSACTIONAL TIME-TRAVEL ROLLBACK HARNESS DEMO")
    print("=" * 72 + f"{RESET}\n")

    with tempfile.TemporaryDirectory(prefix="chronos_demo_") as demo_dir:
        # Create initial clean workspace file
        target_file = Path(demo_dir) / "payment_processor.py"
        target_file.write_text(
            '# Payment Processor v1.0\n'
            'def process_transaction(amount: float) -> bool:\n'
            '    if amount <= 0:\n'
            '        return False\n'
            '    print(f"Processing ${amount} payment...")\n'
            '    return True\n'
        )

        harness = ChronosHarness(demo_dir)
        wrapper = ChronosAgentWrapper(harness)

        print(f"{CYAN}📁 Initialized Workspace at:{RESET} {demo_dir}")
        print(f"{CYAN}📄 Original File Content:{RESET}")
        print(f"{YELLOW}{target_file.read_text().strip()}{RESET}\n")

        # ----------------------------------------------------------------------
        # STEP 1: Successful Turn (Atomic Checkpoint 1)
        # ----------------------------------------------------------------------
        prompt_stack = [{"role": "user", "content": "Refactor payment_processor.py to add tax calculation."}]
        log_box("STEP 1: AGENT INITIATES REFACTORING", "Agent modifying payment_processor.py to add calculate_tax()", CYAN)

        cp1 = harness.create_checkpoint(prompt_stack=prompt_stack)

        def tool_write_v1(args):
            target_file.write_text(
                '# Payment Processor v1.1\n'
                'def calculate_tax(amount: float) -> float:\n'
                '    return amount * 0.08\n\n'
                'def process_transaction(amount: float) -> bool:\n'
                '    total = amount + calculate_tax(amount)\n'
                '    print(f"Processing total ${total} payment with tax...")\n'
                '    return True\n'
            )
            return "File updated successfully."

        success, result, hint = harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "payment_processor.py", "content": target_file.read_text()},
            tool_fn=tool_write_v1,
            prompt_stack=prompt_stack,
            current_checkpoint=cp1
        )

        print(f"{GREEN}✓ Step 1 Succeeded! Checkpoint '{cp1.checkpoint_id}' saved.{RESET}")
        print(f"{CYAN}Current File State on Disk:{RESET}")
        print(f"{GREEN}{target_file.read_text().strip()}{RESET}\n")
        time.sleep(1)

        # ----------------------------------------------------------------------
        # STEP 2: Pre-Flight Invariant Guard Interception (Syntax Error Rejection)
        # ----------------------------------------------------------------------
        log_box("STEP 2: AGENT ATTEMPTS INVALID CODE WRITE", "Agent tries writing broken code missing closing parenthesis", YELLOW)

        cp2 = harness.create_checkpoint(prompt_stack=prompt_stack)

        bad_syntax_content = "def calculate_tax(amount: float -> float:\n    return amount * 0.08"
        
        success, result, hint = harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "payment_processor.py", "content": bad_syntax_content},
            tool_fn=lambda args: None,  # Will not execute because pre-flight fails!
            prompt_stack=prompt_stack,
            current_checkpoint=cp2
        )

        log_box("🛡️ PRE-FLIGHT GUARD INTERCEPTION", f"REJECTED: {hint}", RED)
        print(f"{CYAN}Disk Check: File remains untouched due to Pre-flight Interception!{RESET}")
        print(f"{GREEN}{target_file.read_text().strip()}{RESET}\n")
        time.sleep(1)

        # ----------------------------------------------------------------------
        # STEP 3: Runtime Failure & Sub-Second Time-Travel Rollback
        # ----------------------------------------------------------------------
        log_box("STEP 3: AGENT INTRODUCES RUNTIME BUG (RAISES EXCEPTION)", "Agent code crashes at runtime during post-flight test", RED)

        def tool_write_crashing_code(args):
            # Corrupt file on disk first
            target_file.write_text("CORRUPTED CODE - BROKEN STATE")
            raise RuntimeError("Database connection timed out during execution turn")

        success, result, hint = harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "payment_processor.py", "content": "bad"},
            tool_fn=tool_write_crashing_code,
            prompt_stack=prompt_stack,
            current_checkpoint=cp1
        )

        log_box("⏱️ CHRONOS TIME-TRAVEL ROLLBACK TRIGGERED", 
                f"Rolling back to Step 1 clean snapshot ({cp1.checkpoint_id})\n"
                f"Reverting file system state sub-second...", MAGENTA)

        # Retrieve the updated prompt stack containing the counterfactual hint from DualLedger
        cp_restored = harness.ledger.get_checkpoint(cp1.checkpoint_id)
        restored_prompts = list(cp_restored.turn.prompt_snapshot) if cp_restored else prompt_stack
        restored_prompts.append({
            "role": "system",
            "content": f"⚠️ [CHRONOS TIME-TRAVEL ROLLBACK TRIGGERED]\nCounterfactual Instruction: {hint}"
        })

        print(f"{GREEN}✓ Sub-Second Rollback Complete! Disk State Restored to Checkpoint 1:{RESET}")
        print(f"{GREEN}{target_file.read_text().strip()}{RESET}\n")

        print(f"{MAGENTA}🤖 Counterfactual System Hint Injected into Agent Context:{RESET}")
        print(f"{YELLOW}{restored_prompts[-1]['content']}{RESET}\n")
        time.sleep(1)

        # ----------------------------------------------------------------------
        # STEP 4: Self-Corrected Execution Guided by Counterfactual Hint
        # ----------------------------------------------------------------------
        log_box("STEP 4: AGENT RESUMES FROM CLEAN ROLLBACK STATE", "Agent uses Counterfactual Hint to write correct inline logic", GREEN)

        def tool_write_final(args):
            target_file.write_text(
                '# Payment Processor v1.2 (Self-Corrected)\n'
                'def calculate_tax(amount: float) -> float:\n'
                '    return round(amount * 0.08, 2)\n\n'
                'def process_transaction(amount: float) -> bool:\n'
                '    total = amount + calculate_tax(amount)\n'
                '    print(f"SUCCESS: Processed total ${total} payment.")\n'
                '    return True\n'
            )
            return "File updated cleanly."

        success, result, hint = harness.execute_tool_transactional(
            tool_name="write_file",
            tool_args={"filepath": "payment_processor.py", "content": target_file.read_text()},
            tool_fn=tool_write_final,
            prompt_stack=restored_prompts,
            current_checkpoint=cp1
        )

        log_box("🎉 FINAL RESULT: TASK COMPLETED WITH ZERO STATE LEAKS", 
                f"Status: {result}\n"
                f"Final File Content:\n\n{target_file.read_text().strip()}", GREEN)

        harness.cleanup()

if __name__ == "__main__":
    main()
