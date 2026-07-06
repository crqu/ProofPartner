"""Flatten & Finalize — assembles a complete Lean proof from a proved lemma tree.

Strips unused lemmas and validates final compilation.
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import FLATTEN_PROOF_TEMPLATE
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, NodeStatus
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class FlattenFinalize(BaseAgent):
    """Assembles a proved lemma tree into a self-contained Lean proof."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
    ) -> None:
        super().__init__(name="flatten_finalize", max_retries=2)
        self._llm = llm_client
        self._repl = lean_repl

    def _execute(self, context: AgentContext) -> AgentResult:
        tree_data = context.metadata.get("lemma_tree")
        if not tree_data:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message="No lemma_tree in context metadata",
            )

        tree = LemmaTree.model_validate(tree_data)

        if not tree.all_proved:
            unproved = [
                nid for nid, n in tree.nodes.items() if n.status != NodeStatus.PROVED
            ]
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message=f"Unproved nodes: {unproved}",
            )

        root = tree.nodes[tree.root_id]

        proved_lemmas = self._collect_proved_lemmas(tree)
        root_proof = root.proof_code or ""

        user_content = FLATTEN_PROOF_TEMPLATE.format(
            root_statement=root.statement_lean,
            proved_lemmas=proved_lemmas,
            root_proof=root_proof,
        )

        response = self._llm.complete(
            system="You are an expert Lean 4 programmer. Assemble proofs into a single file.",
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )

        assembled_code = _extract_lean_code(response.content)
        compilation = self._repl.execute(assembled_code)

        if compilation.compilation_status == CompilationStatus.OK and compilation.all_goals_closed:
            log.info("flatten_finalize_success")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                result={"final_proof": assembled_code, "compiles": True},
                token_usage=response.token_usage,
            )

        log.warning(
            "flatten_finalize_compilation_failed",
            errors=compilation.errors,
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILURE,
            result={"final_proof": assembled_code, "compiles": False},
            token_usage=response.token_usage,
            error_message="Assembled proof failed to compile",
        )

    def _collect_proved_lemmas(self, tree: LemmaTree) -> str:
        """Collect proved lemmas in dependency order (leaves first)."""
        parts = []
        for node_id in tree.topological_order:
            node = tree.get_node(node_id)
            if not node or node.node_id == tree.root_id:
                continue
            if node.status == NodeStatus.PROVED and node.proof_code:
                parts.append(f"-- {node.node_id}: {node.statement_nl}\n{node.proof_code}")
        return "\n\n".join(parts) if parts else "-- no sub-lemmas needed"
