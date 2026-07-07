"""Recursive Prover — parent-before-children proving strategy.

Key innovation from the paper:
  1. Prove parent using child lemma statements as sorry premises
  2. Only recurse to prove children after parent validates they're usable
  3. Structured failure diagnosis with reformulation on WeakChildLemma/ContradictoryChild
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    CHILD_REFORMULATION_TEMPLATE,
    FAILURE_DIAGNOSIS_SYSTEM,
    FAILURE_DIAGNOSIS_USER_TEMPLATE,
    PARENT_PROOF_SYSTEM,
    PARENT_PROOF_USER_TEMPLATE,
)
from agentic_research.agents.prover import IterativeProver
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    ProverConfig,
    ProverResult,
    TokenUsage,
)
from agentic_research.models.proof import (
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NodeStatus,
    ProofNode,
    RecursiveProofResult,
)
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_RETRIES_PER_NODE = 3


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class RecursiveProver(BaseAgent):
    """Proves a lemma tree using parent-before-children strategy."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        prover_config: ProverConfig | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_retries_per_node: int = DEFAULT_MAX_RETRIES_PER_NODE,
    ) -> None:
        super().__init__(name="recursive_prover", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._prover_config = prover_config or ProverConfig()
        self._max_depth = max_depth
        self._max_retries_per_node = max_retries_per_node

    def _execute(self, context: AgentContext) -> AgentResult:
        tree_data = context.metadata.get("lemma_tree")
        if not tree_data:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message="No lemma_tree in context metadata",
            )

        tree = LemmaTree.model_validate(tree_data)
        total_tokens = TokenUsage()

        self._prove_node(tree, tree.root_id, total_tokens)

        proved_count = sum(1 for n in tree.nodes.values() if n.status == NodeStatus.PROVED)
        max_depth = max((n.depth for n in tree.nodes.values()), default=0)

        result = RecursiveProofResult(
            root_statement=tree.nodes[tree.root_id].statement_lean,
            proved=tree.all_proved,
            lemma_tree=tree,
            total_nodes=len(tree.nodes),
            proved_nodes=proved_count,
            max_depth_reached=max_depth,
            failure_reason=None if tree.all_proved else "Not all nodes proved",
        )

        status = AgentStatus.SUCCESS if tree.all_proved else AgentStatus.FAILURE
        return AgentResult(
            agent_name=self.name,
            status=status,
            result=result.model_dump(),
            token_usage=total_tokens,
        )

    def _prove_node(self, tree: LemmaTree, node_id: str, tokens: TokenUsage) -> bool:
        node = tree.get_node(node_id)
        if not node:
            return False

        if node.status == NodeStatus.PROVED:
            return True

        if node.depth >= self._max_depth:
            log.warning("recursive_prover_depth_limit", node_id=node_id, depth=node.depth)
            node.status = NodeStatus.FAILED
            node.failure_diagnosis = FailureDiagnosis(
                failure_type=FailureType.STUCK_GOAL,
                description=f"Depth limit {self._max_depth} reached",
            )
            return False

        children = tree.get_children(node_id)
        if not children:
            return self._prove_leaf(tree, node, tokens)

        return self._prove_parent(tree, node, tokens)

    def _prove_leaf(self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage) -> bool:
        """Prove a leaf node directly using the iterative prover."""
        log.info("recursive_prover_leaf", node_id=node.node_id)

        prover = IterativeProver(
            llm_client=self._llm,
            lean_repl=self._repl,
            config=self._prover_config,
        )
        ctx = AgentContext(task=node.statement_lean)
        result = prover.run(ctx)

        tokens.input_tokens += result.token_usage.input_tokens
        tokens.output_tokens += result.token_usage.output_tokens

        if result.result:
            prover_result = ProverResult.model_validate(result.result)
            if prover_result.proved:
                node.status = NodeStatus.PROVED
                node.proof_code = prover_result.final_proof
                return True

        node.status = NodeStatus.FAILED
        node.failure_diagnosis = FailureDiagnosis(
            failure_type=FailureType.STUCK_GOAL,
            description="Iterative prover exhausted",
        )
        return False

    def _prove_parent(self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage) -> bool:
        """Parent-before-children: prove parent using child sorry-premises, then recurse."""
        for retry in range(self._max_retries_per_node):
            node.retries_used = retry + 1

            parent_proved = self._prove_parent_with_children(tree, node, tokens)

            if not parent_proved:
                diagnosis = self._diagnose_failure(tree, node, tokens)
                node.failure_diagnosis = diagnosis

                if diagnosis and diagnosis.failure_type in (
                    FailureType.WEAK_CHILD_LEMMA,
                    FailureType.CONTRADICTORY_CHILD,
                ) and diagnosis.problematic_child_id:
                    reformulated = self._reformulate_child(
                        tree, node, diagnosis, tokens
                    )
                    if reformulated:
                        log.info(
                            "recursive_prover_reformulated",
                            node_id=node.node_id,
                            child_id=diagnosis.problematic_child_id,
                            retry=retry + 1,
                        )
                        continue

                log.info(
                    "recursive_prover_parent_failed",
                    node_id=node.node_id,
                    failure_type=diagnosis.failure_type.value if diagnosis else "unknown",
                    retry=retry + 1,
                )
                node.status = NodeStatus.FAILED
                return False

            all_children_proved = True
            for child in tree.get_children(node.node_id):
                if child.status == NodeStatus.PROVED:
                    continue
                if not self._prove_node(tree, child.node_id, tokens):
                    all_children_proved = False
                    break

            if all_children_proved:
                node.status = NodeStatus.PROVED
                return True

            log.info(
                "recursive_prover_children_failed",
                node_id=node.node_id,
                retry=retry + 1,
            )

        node.status = NodeStatus.FAILED
        return False

    def _prove_parent_with_children(
        self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage
    ) -> bool:
        """Prove the parent assuming child lemmas are true (sorry)."""
        children = tree.get_children(node.node_id)
        child_decls = "\n\n".join(
            c.statement_lean for c in children if c.statement_lean
        )

        if not child_decls:
            return False

        user_content = PARENT_PROOF_USER_TEMPLATE.format(
            parent_statement=node.statement_lean,
            child_declarations=child_decls,
        )

        response = self._llm.complete(
            system=PARENT_PROOF_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )
        tokens.input_tokens += response.token_usage.input_tokens
        tokens.output_tokens += response.token_usage.output_tokens

        proof_code = _extract_lean_code(response.content)
        compilation = self._repl.execute(proof_code)

        if compilation.compilation_status == CompilationStatus.OK:
            node.proof_code = proof_code
            return True

        return False

    def _diagnose_failure(
        self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage
    ) -> FailureDiagnosis:
        """Use LLM to classify why the parent proof failed."""
        children = tree.get_children(node.node_id)
        child_decls = "\n\n".join(
            f"-- {c.node_id}\n{c.statement_lean}" for c in children if c.statement_lean
        )

        user_content = FAILURE_DIAGNOSIS_USER_TEMPLATE.format(
            parent_statement=node.statement_lean,
            child_declarations=child_decls,
            failed_proof=node.proof_code or "-- no proof attempt",
            errors="Parent proof did not compile or close all goals",
        )

        response = self._llm.complete(
            system=FAILURE_DIAGNOSIS_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )
        tokens.input_tokens += response.token_usage.input_tokens
        tokens.output_tokens += response.token_usage.output_tokens

        parsed = self._llm.extract_json(response.content)
        if isinstance(parsed, dict):
            try:
                ft = FailureType(parsed.get("failure_type", "stuck_goal"))
            except ValueError:
                ft = FailureType.STUCK_GOAL
            return FailureDiagnosis(
                failure_type=ft,
                description=parsed.get("description", ""),
                problematic_child_id=parsed.get("problematic_child_id"),
                suggested_fix=parsed.get("suggested_fix", ""),
            )

        return FailureDiagnosis(
            failure_type=FailureType.STUCK_GOAL,
            description="Could not parse failure diagnosis",
        )

    def _reformulate_child(
        self,
        tree: LemmaTree,
        parent_node: ProofNode,
        diagnosis: FailureDiagnosis,
        tokens: TokenUsage,
    ) -> bool:
        """Reformulate a problematic child lemma and re-leanify it."""
        child_id = diagnosis.problematic_child_id
        if not child_id:
            return False
        child = tree.get_node(child_id)
        if not child:
            return False

        user_content = CHILD_REFORMULATION_TEMPLATE.format(
            parent_statement=parent_node.statement_lean,
            child_id=child_id,
            child_statement_nl=child.statement_nl,
            child_statement_lean=child.statement_lean,
            failure_type=diagnosis.failure_type.value,
            failure_description=diagnosis.description,
            suggested_fix=diagnosis.suggested_fix,
        )

        response = self._llm.complete(
            system=FAILURE_DIAGNOSIS_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )
        tokens.input_tokens += response.token_usage.input_tokens
        tokens.output_tokens += response.token_usage.output_tokens

        parsed = self._llm.extract_json(response.content)
        if isinstance(parsed, dict) and "reformulated_statement" in parsed:
            child.statement_nl = parsed["reformulated_statement"]
            child.statement_lean = ""
            child.status = NodeStatus.REFORMULATED
            child.proof_code = None
            return True

        return False
