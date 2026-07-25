"""
Causal Execution Trace Graph (CTG) for Chronos-Agent.

Every tool call, model decision, file mutation, and failure event
gets recorded as a typed CausalNode and linked to the event that
caused it via a directed CausalEdge — forming a live, queryable DAG.

This enables surgical root cause identification:
  "test_suite FAILED because calculate_tax() returned None
   because amount was never validated at step 3."
"""

import time
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple


class NodeKind(str, Enum):
    """The semantic type of a causal node."""
    AGENT_DECISION  = "AGENT_DECISION"   # LLM chose to call a tool
    TOOL_CALL       = "TOOL_CALL"        # A tool was invoked
    FILE_MUTATION   = "FILE_MUTATION"    # A file was created/modified/deleted
    INVARIANT_CHECK = "INVARIANT_CHECK"  # A pre/post flight guard ran
    ROLLBACK        = "ROLLBACK"         # A time-travel rollback occurred
    COUNTERFACTUAL  = "COUNTERFACTUAL"   # A corrective hint was injected
    CHECKPOINT      = "CHECKPOINT"       # An atomic snapshot was taken
    FAILURE         = "FAILURE"          # A failure event was recorded


class NodeStatus(str, Enum):
    SUCCESS  = "SUCCESS"
    FAILURE  = "FAILURE"
    PENDING  = "PENDING"
    REVERTED = "REVERTED"


class CausalNode:
    """
    A single event node in the Causal Execution Trace Graph.
    Represents one atomic action, decision, or state change.
    """
    def __init__(
        self,
        node_id: str,
        kind: NodeKind,
        label: str,
        step: int,
        status: NodeStatus = NodeStatus.PENDING,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ):
        self.node_id = node_id
        self.kind = kind
        self.label = label
        self.step = step
        self.status = status
        self.metadata = metadata or {}
        self.checkpoint_id = checkpoint_id
        self.timestamp = time.time()

    def mark_success(self, result_summary: Optional[str] = None):
        self.status = NodeStatus.SUCCESS
        if result_summary:
            self.metadata["result_summary"] = result_summary

    def mark_failure(self, error: str, hint: Optional[str] = None):
        self.status = NodeStatus.FAILURE
        self.metadata["error"] = error
        if hint:
            self.metadata["hint"] = hint

    def mark_reverted(self):
        self.status = NodeStatus.REVERTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "label": self.label,
            "step": self.step,
            "status": self.status.value,
            "metadata": self.metadata,
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
        }

    def __repr__(self):
        return f"<CausalNode [{self.kind.value}] '{self.label}' step={self.step} status={self.status.value}>"


class CausalEdge:
    """
    A directed causal edge: cause_node → effect_node.
    Records WHY each event happened.
    """
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        relation: str = "caused",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.relation = relation           # e.g. "caused", "triggered_rollback", "corrected_by"
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_node_id,
            "to": self.to_node_id,
            "relation": self.relation,
            "metadata": self.metadata,
        }


class CausalTraceGraph:
    """
    The Live Causal Execution Trace Graph (CTG).

    A directed acyclic graph (DAG) tracking every action and its causal lineage.
    Enables surgical root-cause analysis instead of blind log scanning.

    Usage:
        ctg = CausalTraceGraph()
        n1 = ctg.record_decision("Agent decided to call write_file()", step=1)
        n2 = ctg.record_tool_call("write_file", args={"path":"app.py"}, step=1, caused_by=n1)
        n3 = ctg.record_failure("SyntaxError in app.py", step=1, caused_by=n2)
        path = ctg.trace_root_cause(n3.node_id)
        # Returns: [decision → write_file → SyntaxError]
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self._adjacency_out: Dict[str, List[str]] = {}  # node -> [children]
        self._adjacency_in:  Dict[str, List[str]] = {}  # node -> [parents]
        self._node_counter = 0

    # ------------------------------------------------------------------
    # Node Registration API
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}_{int(time.time() * 1000)}"

    def _add_node(self, node: CausalNode, caused_by: Optional[CausalNode] = None, relation: str = "caused") -> CausalNode:
        self.nodes[node.node_id] = node
        self._adjacency_out.setdefault(node.node_id, [])
        self._adjacency_in.setdefault(node.node_id, [])
        if caused_by:
            self._add_edge(from_node=caused_by, to_node=node, relation=relation)
        return node

    def _add_edge(self, from_node: CausalNode, to_node: CausalNode, relation: str = "caused") -> CausalEdge:
        edge = CausalEdge(from_node.node_id, to_node.node_id, relation)
        self.edges.append(edge)
        self._adjacency_out.setdefault(from_node.node_id, []).append(to_node.node_id)
        self._adjacency_in.setdefault(to_node.node_id, []).append(from_node.node_id)
        return edge

    def record_checkpoint(self, checkpoint_id: str, step: int, caused_by: Optional[CausalNode] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("chk"),
            kind=NodeKind.CHECKPOINT,
            label=f"Checkpoint {checkpoint_id}",
            step=step,
            status=NodeStatus.SUCCESS,
            checkpoint_id=checkpoint_id
        )
        return self._add_node(node, caused_by)

    def record_decision(self, label: str, step: int, metadata: Optional[Dict] = None, caused_by: Optional[CausalNode] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("dec"),
            kind=NodeKind.AGENT_DECISION,
            label=label,
            step=step,
            status=NodeStatus.PENDING,
            metadata=metadata or {}
        )
        return self._add_node(node, caused_by)

    def record_tool_call(self, tool_name: str, args: Dict[str, Any], step: int, caused_by: Optional[CausalNode] = None) -> CausalNode:
        safe_args = {k: str(v)[:120] for k, v in args.items()}  # Truncate large args
        node = CausalNode(
            node_id=self._next_id("tool"),
            kind=NodeKind.TOOL_CALL,
            label=f"tool:{tool_name}",
            step=step,
            status=NodeStatus.PENDING,
            metadata={"tool_name": tool_name, "args": safe_args}
        )
        return self._add_node(node, caused_by)

    def record_file_mutation(self, filepath: str, mutation_type: str, step: int, caused_by: Optional[CausalNode] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("fs"),
            kind=NodeKind.FILE_MUTATION,
            label=f"{mutation_type}:{filepath}",
            step=step,
            status=NodeStatus.PENDING,
            metadata={"filepath": filepath, "mutation_type": mutation_type}
        )
        return self._add_node(node, caused_by)

    def record_invariant_check(self, rule_name: str, passed: bool, step: int, caused_by: Optional[CausalNode] = None, error: Optional[str] = None, hint: Optional[str] = None) -> CausalNode:
        status = NodeStatus.SUCCESS if passed else NodeStatus.FAILURE
        meta = {"rule": rule_name}
        if error:
            meta["error"] = error
        if hint:
            meta["hint"] = hint
        node = CausalNode(
            node_id=self._next_id("inv"),
            kind=NodeKind.INVARIANT_CHECK,
            label=f"invariant:{rule_name}",
            step=step,
            status=status,
            metadata=meta
        )
        return self._add_node(node, caused_by)

    def record_failure(self, reason: str, step: int, caused_by: Optional[CausalNode] = None, hint: Optional[str] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("fail"),
            kind=NodeKind.FAILURE,
            label=f"FAILURE: {reason[:80]}",
            step=step,
            status=NodeStatus.FAILURE,
            metadata={"reason": reason, "hint": hint or ""}
        )
        return self._add_node(node, caused_by, relation="caused_failure")

    def record_rollback(self, to_checkpoint_id: str, step: int, caused_by: Optional[CausalNode] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("rb"),
            kind=NodeKind.ROLLBACK,
            label=f"ROLLBACK → {to_checkpoint_id}",
            step=step,
            status=NodeStatus.SUCCESS,
            checkpoint_id=to_checkpoint_id
        )
        # Mark caused_by node as reverted
        if caused_by:
            caused_by.mark_reverted()
        return self._add_node(node, caused_by, relation="triggered_rollback")

    def record_counterfactual(self, hint: str, step: int, caused_by: Optional[CausalNode] = None) -> CausalNode:
        node = CausalNode(
            node_id=self._next_id("cf"),
            kind=NodeKind.COUNTERFACTUAL,
            label=f"HINT: {hint[:80]}",
            step=step,
            status=NodeStatus.SUCCESS,
            metadata={"hint": hint}
        )
        return self._add_node(node, caused_by, relation="corrected_by")

    # ------------------------------------------------------------------
    # Root Cause Analysis API
    # ------------------------------------------------------------------

    def trace_root_cause(self, failure_node_id: str) -> List[CausalNode]:
        """
        Traces backwards from a failure node to find the root cause chain.
        Returns the ordered causal path: [root_cause, ..., failure_node].
        """
        if failure_node_id not in self.nodes:
            return []

        path = []
        visited: Set[str] = set()
        current_id = failure_node_id

        while current_id:
            if current_id in visited:
                break
            visited.add(current_id)
            path.append(self.nodes[current_id])
            parents = self._adjacency_in.get(current_id, [])
            current_id = parents[0] if parents else None

        path.reverse()
        return path

    def find_all_failures(self) -> List[CausalNode]:
        """Returns all FAILURE nodes in the graph."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILURE]

    def find_reverted_nodes(self) -> List[CausalNode]:
        """Returns all nodes that were reverted by a rollback."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.REVERTED]

    def explain_failure(self, failure_node_id: str) -> str:
        """
        Generates a human-readable causal chain explanation:
        "FAILURE because step_X caused Y because step_N caused Z"
        """
        chain = self.trace_root_cause(failure_node_id)
        if not chain:
            return "No causal chain found."

        parts = []
        for i, node in enumerate(chain):
            if i == 0:
                parts.append(f"Root cause at Step {node.step}: [{node.kind.value}] '{node.label}'")
            elif i == len(chain) - 1:
                parts.append(f"→ which ultimately caused Step {node.step}: [{node.kind.value}] '{node.label}'")
            else:
                parts.append(f"→ which caused Step {node.step}: [{node.kind.value}] '{node.label}'")

        hint = chain[-1].metadata.get("hint") or chain[0].metadata.get("hint")
        if hint:
            parts.append(f"\n💡 Fix: {hint}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the full graph to a JSON-compatible dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "failures": len(self.find_all_failures()),
                "reverted": len(self.find_reverted_nodes()),
            }
        }

    def to_mermaid(self) -> str:
        """
        Renders the live CTG as a Mermaid flowchart for visual inspection.
        Useful for embedding into logs, dashboards, or GitHub Reports.
        """
        KIND_EMOJI = {
            NodeKind.AGENT_DECISION:  "🧠",
            NodeKind.TOOL_CALL:       "🛠️",
            NodeKind.FILE_MUTATION:   "📁",
            NodeKind.INVARIANT_CHECK: "🛡️",
            NodeKind.ROLLBACK:        "⏱️",
            NodeKind.COUNTERFACTUAL:  "💡",
            NodeKind.CHECKPOINT:      "🔒",
            NodeKind.FAILURE:         "❌",
        }
        STATUS_STYLE = {
            NodeStatus.SUCCESS:  "fill:#d1fae5,stroke:#059669",
            NodeStatus.FAILURE:  "fill:#ffe4e6,stroke:#e11d48",
            NodeStatus.REVERTED: "fill:#fef3c7,stroke:#d97706",
            NodeStatus.PENDING:  "fill:#e0f2fe,stroke:#0284c7",
        }

        lines = ["graph TD"]
        style_lines = []

        for node in self.nodes.values():
            safe_id = node.node_id.replace("-", "_")
            emoji = KIND_EMOJI.get(node.kind, "⬜")
            label = node.label.replace('"', "'")
            lines.append(f'    {safe_id}["{emoji} Step {node.step}<br/>{label}"]')
            style = STATUS_STYLE.get(node.status, "")
            if style:
                style_lines.append(f"    style {safe_id} {style},color:#0f172a")

        for edge in self.edges:
            from_id = edge.from_node_id.replace("-", "_")
            to_id   = edge.to_node_id.replace("-", "_")
            lines.append(f"    {from_id} -->|{edge.relation}| {to_id}")

        lines.extend(style_lines)
        return "\n".join(lines)
