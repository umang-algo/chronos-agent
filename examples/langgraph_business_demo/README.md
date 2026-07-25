# Chronos + LangGraph Business Agent Demonstration ⚡

This example demonstrates **Chronos-Agent** protecting a **LangGraph E-Commerce Refund & Financial Audit Agent** against financial state corruption, unauthorized mutations, and runtime database crashes.

---

## 🏗️ What This Architecture Demonstrates

1. **Scope-Locked Policy Enforcement**: Blocks unauthorized tool execution and restrict file paths to `mock_db/*` using role RBAC rules (`policy_rules.py`).
2. **Financial Invariants**: Rejects refund requests exceeding original order amounts (`check_refund_invariant`).
3. **Sub-Second Time-Travel Rollback**: Auto-reverts corrupted files on disk in ~10ms when a database connection timeout or runtime exception occurs.
4. **Causal Trace Graph (CTG) Root Cause Explanation**: Pinpoints exact causal chain leading to a failure for LLM self-correction.

---

## 🚀 How to Run

From the root of the repository:

```bash
python3 examples/langgraph_business_demo/demo_run.py
```

---

## 🧪 Scenarios Covered in the Demo

- **Scenario 1 (Valid Refund)**: Order #1001 inspected ($150 total), $50 refund processed cleanly. Checkpoint created.
- **Scenario 2 (Illegal Refund Rejection)**: Agent attempts to process $5,000 refund on Order #1002 ($45.50 total). Chronos pre-flight invariant engine blocks execution before the ledger is modified.
- **Scenario 3 (DB Crash & Sub-Second Rollback)**: Tool execution crashes mid-write (simulating network timeout). Chronos instantly restores `mock_db/financial_ledger.json` to clean Scenario 1 state and prints the live CTG causal failure chain.
