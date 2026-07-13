"""Recursive Prover — parent-before-children proving strategy.

Key innovation from the paper:
  1. Prove parent using child lemma statements as sorry premises
  2. Only recurse to prove children after parent validates they're usable
  3. Structured failure diagnosis with reformulation on WeakChildLemma/ContradictoryChild
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

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
    ErrorCategory,
    FailureDiagnosis,
    FailureType,
    LemmaTree,
    NodeStatus,
    ProofNode,
    RecursiveProofResult,
)
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

if TYPE_CHECKING:
    from agentic_research.agents.lemma_breakdown import LemmaBreakdown
    from agentic_research.agents.lemma_leanifier import LemmaLeanifier
    from agentic_research.agents.nl_prover import NaturalLanguageProver
    from agentic_research.agents.proof_corrector import ProofCorrector
    from agentic_research.agents.proof_detailer import ProofDetailer

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

    _MAX_TOTAL_NODES = 50

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        prover_config: ProverConfig | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_retries_per_node: int = DEFAULT_MAX_RETRIES_PER_NODE,
        lean_preamble: str | None = None,
        leanifier: LemmaLeanifier | None = None,
        nl_prover: NaturalLanguageProver | None = None,
        proof_detailer: ProofDetailer | None = None,
        breakdown: LemmaBreakdown | None = None,
        proof_corrector: ProofCorrector | None = None,
    ) -> None:
        super().__init__(name="recursive_prover", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._prover_config = prover_config or ProverConfig()
        self._max_depth = max_depth
        self._max_retries_per_node = max_retries_per_node
        self._lean_preamble = lean_preamble
        self._leanifier = leanifier
        self._nl_prover = nl_prover
        self._proof_detailer = proof_detailer
        self._breakdown = breakdown
        self._proof_corrector = proof_corrector
        self._total_nodes = 0

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
        self._total_nodes = len(tree.nodes)

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

        if node.from_prior_work:
            node.status = NodeStatus.PROVED
            node.proof_code = node.statement_lean
            log.info("recursive_prover_axiom_skip", node_id=node_id)
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

        # H5: Try ProofCorrector for structured error diagnosis
        if self._proof_corrector and result.result:
            prover_result = ProverResult.model_validate(result.result)
            error_msg = prover_result.failure_reason or "Proof search exhausted"
            if "timeout" not in error_msg.lower():
                correction = self._proof_corrector.correct(
                    failed_proof=prover_result.final_proof or node.statement_lean,
                    error_message=error_msg,
                    lean_goal_state=node.statement_lean,
                )
                tokens.input_tokens += self._proof_corrector.cumulative_tokens.input_tokens
                tokens.output_tokens += self._proof_corrector.cumulative_tokens.output_tokens

                if correction.error_category != ErrorCategory.TIMEOUT and correction.suggested_tactics:
                    correction_hint = (
                        f"\n\n[Correction context]\n"
                        f"Error: {correction.error_category.value}\n"
                        f"Suggested tactics: {', '.join(correction.suggested_tactics)}\n"
                        f"Revised sketch:\n{correction.revised_proof_sketch}"
                    )
                    retry_ctx = AgentContext(
                        task=node.statement_lean + correction_hint
                    )
                    retry_prover = IterativeProver(
                        llm_client=self._llm,
                        lean_repl=self._repl,
                        config=self._prover_config,
                    )
                    retry_result = retry_prover.run(retry_ctx)
                    tokens.input_tokens += retry_result.token_usage.input_tokens
                    tokens.output_tokens += retry_result.token_usage.output_tokens

                    if retry_result.result:
                        retry_prover_result = ProverResult.model_validate(retry_result.result)
                        if retry_prover_result.proved:
                            node.status = NodeStatus.PROVED
                            node.proof_code = retry_prover_result.final_proof
                            log.info("recursive_prover_correction_success", node_id=node.node_id)
                            return True

                    node.failure_diagnosis = FailureDiagnosis(
                        failure_type=FailureType.STUCK_GOAL,
                        description=f"Correction retry failed ({correction.error_category.value})",
                    )

        # H4: Try recursive decomposition of stuck leaf
        if self._decompose_stuck_leaf(tree, node, tokens):
            return self._prove_parent(tree, node, tokens)

        node.status = NodeStatus.FAILED
        if not node.failure_diagnosis:
            node.failure_diagnosis = FailureDiagnosis(
                failure_type=FailureType.STUCK_GOAL,
                description="Iterative prover exhausted",
            )
        return False

    def _prove_parent(self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage) -> bool:
        """Parent-before-children: prove parent using child sorry-premises, then recurse."""
        nl_context, _ = self._generate_nl_context(node, tokens)

        previous_proof: str | None = None
        previous_errors: str = ""
        for retry in range(self._max_retries_per_node):
            node.retries_used = retry + 1

            parent_proved, attempt_code, attempt_errors = (
                self._prove_parent_with_children(
                    tree, node, tokens,
                    previous_proof=previous_proof,
                    previous_errors=previous_errors,
                    nl_context=nl_context,
                )
            )

            if not parent_proved:
                previous_proof = attempt_code
                previous_errors = attempt_errors

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
                continue

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

    @staticmethod
    def _format_child_declaration(child: ProofNode) -> str:
        """Format a child node as an axiom declaration with usage hint.

        All children are declared as axioms (not theorems with sorry) so the
        parent proof compiles without sorry warnings.
        """
        stmt = child.statement_lean.strip()
        name = child.node_id.replace("-", "_")
        if stmt.startswith("axiom "):
            pass
        elif stmt.startswith("theorem ") or stmt.startswith("lemma "):
            sig = re.sub(r"^(?:theorem|lemma)\s+\S+", "", stmt)
            sig = re.sub(r"\s*:=\s*(?:by\s+)?sorry\s*$", "", sig)
            stmt = f"axiom {name}{sig}"
        else:
            stmt = f"axiom {name} : {stmt}"
        return f"{stmt}\n-- Use: have <result> := {name} <args>"

    def _prove_parent_with_children(
        self,
        tree: LemmaTree,
        node: ProofNode,
        tokens: TokenUsage,
        previous_proof: str | None = None,
        previous_errors: str = "",
        nl_context: str = "",
    ) -> tuple[bool, str, str]:
        """Prove the parent assuming child lemmas are true (sorry).

        Returns (proved, proof_code_attempted, compilation_errors).
        """
        children = tree.get_children(node.node_id)
        child_decls = "\n\n".join(
            self._format_child_declaration(c)
            for c in children if c.statement_lean
        )

        if not child_decls:
            return False, "", ""

        user_content = PARENT_PROOF_USER_TEMPLATE.format(
            parent_statement=node.statement_lean,
            child_declarations=child_decls,
        )

        if nl_context:
            user_content += f"\n\n{nl_context}"

        if previous_proof and previous_errors:
            user_content += (
                "\n\n## Previous Attempt (FAILED — do NOT repeat)\n"
                f"```lean\n{previous_proof}\n```\n\n"
                f"## Compilation Errors\n{previous_errors}\n\n"
                "Fix the errors above. Try a DIFFERENT proof approach."
            )

        response = self._llm.complete(
            system=PARENT_PROOF_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            use_extended_thinking=self._prover_config.use_extended_thinking,
            use_cache=True,
        )
        tokens.input_tokens += response.token_usage.input_tokens
        tokens.output_tokens += response.token_usage.output_tokens

        proof_code = _extract_lean_code(response.content)
        compile_code = (self._lean_preamble + "\n\n" + proof_code) if self._lean_preamble else proof_code
        compilation = self._repl.execute(compile_code)

        errors_str = "\n".join(compilation.errors or [])
        uses_sorry = any('sorry' in w for w in (compilation.warnings or []))
        if compilation.compilation_status == CompilationStatus.OK and not uses_sorry:
            node.proof_code = proof_code
            return True, proof_code, ""

        return False, proof_code, errors_str

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

            self._leanify_reformulated_node(tree, child, tokens)
            return True

        return False

    def _leanify_reformulated_node(
        self, tree: LemmaTree, child: ProofNode, tokens: TokenUsage
    ) -> bool:
        """Re-leanify a reformulated child so it can participate in parent proofs."""
        if not self._leanifier:
            return False

        parent = tree.get_node(child.parent_id) if child.parent_id else None
        parent_stmt = parent.statement_lean if parent else ""
        siblings = self._get_sibling_statements(tree, child)

        lean_code, usage = self._leanifier.leanify_single_node(
            child, parent_stmt, siblings
        )
        tokens.input_tokens += usage.input_tokens
        tokens.output_tokens += usage.output_tokens

        if lean_code:
            child.statement_lean = lean_code
            child.status = NodeStatus.PENDING
            log.info("reformulated_child_releanified", node_id=child.node_id)
            return True

        log.warning("reformulated_child_leanify_failed", node_id=child.node_id)
        return False

    def _get_sibling_statements(self, tree: LemmaTree, node: ProofNode) -> str:
        """Get Lean statements of sibling nodes."""
        if not node.parent_id:
            return ""
        parent = tree.get_node(node.parent_id)
        if not parent:
            return ""
        parts = []
        for cid in parent.children:
            if cid == node.node_id:
                continue
            sibling = tree.get_node(cid)
            if sibling and sibling.statement_lean:
                parts.append(f"-- {cid}\n{sibling.statement_lean}")
        return "\n\n".join(parts)

    def _generate_nl_context(
        self, node: ProofNode, tokens: TokenUsage
    ) -> tuple[str, str]:
        """Generate NL proof sketch and tactic hints for a node.

        Returns (nl_context_string, tactic_hints). No-ops if nl_prover is None.
        """
        if not self._nl_prover:
            return "", ""

        sketch, gen_tokens = self._nl_prover.generate_proof(
            statement=node.statement_lean,
            statement_nl=node.statement_nl or None,
        )
        tokens.input_tokens += gen_tokens.input_tokens
        tokens.output_tokens += gen_tokens.output_tokens

        if not sketch.proof_steps:
            return "", ""

        tactic_hints = ""
        if self._proof_detailer:
            tactic_hints = self._proof_detailer.detail_sketch(sketch)
            tokens.input_tokens += self._proof_detailer.cumulative_tokens.input_tokens
            tokens.output_tokens += self._proof_detailer.cumulative_tokens.output_tokens

        nl_context = (
            f"## NL Proof Context\n"
            f"Strategy: {sketch.overall_strategy}\n"
        )
        for i, step in enumerate(sketch.proof_steps, 1):
            nl_context += f"Step {i}: {step.claim} — {step.reasoning}\n"
        if tactic_hints:
            nl_context += f"\n## Tactic Hints\n{tactic_hints}\n"

        log.info(
            "recursive_prover_nl_context",
            node_id=node.node_id,
            steps=len(sketch.proof_steps),
        )
        return nl_context, tactic_hints

    _MAX_DECOMPOSITION_CHILDREN = 5

    def _decompose_stuck_leaf(
        self, tree: LemmaTree, node: ProofNode, tokens: TokenUsage
    ) -> bool:
        """Decompose a stuck leaf into sub-lemmas via breakdown.

        Returns True if decomposition succeeded and the node now has children.
        """
        if not self._breakdown:
            return False
        if node.depth >= self._max_depth - 1:
            return False
        if self._total_nodes >= self._MAX_TOTAL_NODES:
            log.warning("recursive_prover_node_cap", total=self._total_nodes)
            return False

        nl_context, tactic_hints = self._generate_nl_context(node, tokens)

        metadata: dict = {"statement_lean": node.statement_lean}
        if nl_context:
            metadata["nl_context"] = nl_context
        if tactic_hints:
            metadata["tactic_hints"] = tactic_hints

        ctx = AgentContext(
            task=node.statement_nl or node.statement_lean,
            metadata=metadata,
        )
        result = self._breakdown.run(ctx)
        tokens.input_tokens += self._breakdown.cumulative_tokens.input_tokens
        tokens.output_tokens += self._breakdown.cumulative_tokens.output_tokens

        if result.status != AgentStatus.SUCCESS or not result.result:
            return False

        sub_tree = LemmaTree.model_validate(result.result)
        child_nodes = [
            n for nid, n in sub_tree.nodes.items() if nid != sub_tree.root_id
        ]

        if not child_nodes or len(child_nodes) > self._MAX_DECOMPOSITION_CHILDREN:
            log.info(
                "recursive_prover_decomp_skip",
                node_id=node.node_id,
                children=len(child_nodes),
            )
            return False

        new_children: list[str] = []
        for child in child_nodes:
            child_id = f"{node.node_id}_{child.node_id}"
            new_node = ProofNode(
                node_id=child_id,
                statement_nl=child.statement_nl,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )

            if self._leanifier:
                lean_code, usage = self._leanifier.leanify_single_node(
                    new_node, node.statement_lean
                )
                tokens.input_tokens += usage.input_tokens
                tokens.output_tokens += usage.output_tokens
                if lean_code:
                    new_node.statement_lean = lean_code

            tree.nodes[child_id] = new_node
            new_children.append(child_id)
            self._total_nodes += 1

        node.children = new_children
        node.status = NodeStatus.PENDING
        node.failure_diagnosis = None
        node.proof_code = None

        tree.topological_order = new_children + tree.topological_order

        log.info(
            "recursive_prover_leaf_decomposed",
            node_id=node.node_id,
            new_children=len(new_children),
            total_nodes=self._total_nodes,
        )
        return True
