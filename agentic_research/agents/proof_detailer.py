"""Proof Detailer — expands coarse NL proof steps to tactic-sized sketches.

Selectively details complex nodes based on a complexity heuristic,
skipping axioms, trivial leaves, and already-detailed nodes.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    DETAIL_SKETCH_SYSTEM,
    DETAIL_SKETCH_USER_TEMPLATE,
    PROOF_DETAILER_SYSTEM,
    PROOF_DETAILER_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, NLProofSketch, ProofNode

log = get_logger(__name__)

COMPLEXITY_THRESHOLD = 0
COMPLEXITY_CUE_PHRASES = [
    "clearly", "obviously", "trivially", "by standard arguments",
    "it is well known", "straightforward", "immediate",
]
COMPLEXITY_OPERATORS = [
    "integral", "limit", "sup", "inf", "lim sup", "lim inf",
    "measure", "expectation", "probability",
]


def compute_complexity_score(node: ProofNode) -> int:
    """Score a node's complexity to decide if it needs detailing."""
    score = 0

    nl_len = len(node.statement_nl)
    score += min(30, nl_len // 5)

    lean_len = len(node.statement_lean) if node.statement_lean else 0
    score += min(20, lean_len // 10)

    text = node.statement_nl.lower()
    quantifier_markers = ["for all", "for every", "there exists", "∀", "∃"]
    nesting = sum(1 for m in quantifier_markers if m in text)
    score += min(25, nesting * 8)

    for op in COMPLEXITY_OPERATORS:
        if op in text:
            score += 5
    for phrase in COMPLEXITY_CUE_PHRASES:
        if phrase in text:
            score += 5
            break

    return score


class ProofDetailer(BaseAgent):
    """Expands coarse NL lemma statements into tactic-granularity proof sketches."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        complexity_threshold: int = COMPLEXITY_THRESHOLD,
    ) -> None:
        super().__init__(name="proof_detailer", max_retries=1)
        self._llm = llm_client
        self._complexity_threshold = complexity_threshold

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
        detailed_count = 0

        for node_id in tree.topological_order:
            node = tree.get_node(node_id)
            if not node:
                continue

            if not self._should_detail(node, tree):
                continue

            sketch, tokens = self._detail_node(node, tree)
            total_tokens.input_tokens += tokens.input_tokens
            total_tokens.output_tokens += tokens.output_tokens

            if sketch:
                node.proof_sketch_nl = sketch
                detailed_count += 1
                log.info("proof_detailer_detailed", node_id=node_id)

        log.info("proof_detailer_done", detailed=detailed_count)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=tree.model_dump(),
            token_usage=total_tokens,
        )

    def _should_detail(self, node: ProofNode, tree: LemmaTree) -> bool:
        if node.from_prior_work:
            return False

        if node.proof_sketch_nl:
            return False

        if node.node_id == tree.root_id:
            return False

        score = compute_complexity_score(node)
        return score >= self._complexity_threshold

    def _detail_node(
        self, node: ProofNode, tree: LemmaTree
    ) -> tuple[str | None, TokenUsage]:
        parent = tree.get_node(node.parent_id) if node.parent_id else None
        parent_stmt = parent.statement_nl if parent else "-- (root)"

        user_content = PROOF_DETAILER_USER_TEMPLATE.format(
            node_id=node.node_id,
            statement_nl=node.statement_nl,
            depth=node.depth,
            statement_lean=node.statement_lean or "-- not yet formalized",
            parent_statement=parent_stmt,
        )

        response = self._llm.complete(
            system=PROOF_DETAILER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        sketch = self._parse_sketch(response.content)
        return sketch, response.token_usage

    def detail_sketch(self, sketch: NLProofSketch) -> str:
        """Expand NL proof steps into tactic-level hints."""
        step_lines = []
        for i, step in enumerate(sketch.proof_steps, 1):
            parts = [f"Step {i}: {step.claim}"]
            parts.append(f"  Reasoning: {step.reasoning}")
            for sc in step.sub_claims:
                parts.append(f"  - Sub-claim: {sc}")
            step_lines.append("\n".join(parts))

        user_content = DETAIL_SKETCH_USER_TEMPLATE.format(
            overall_strategy=sketch.overall_strategy,
            proof_steps="\n\n".join(step_lines),
        )

        response = self._llm.complete(
            system=DETAIL_SKETCH_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        log.info(
            "proof_detailer_sketch_detailed",
            steps=len(sketch.proof_steps),
            response_len=len(response.content),
        )
        return response.content.strip()

    def _parse_sketch(self, response_text: str) -> str | None:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            return None

        if not parsed.get("needs_detailing", False):
            return None

        steps = parsed.get("proof_sketch", [])
        if not steps:
            return None

        lines = []
        for step in steps:
            num = step.get("step_number", "?")
            claim = step.get("claim", "")
            justification = step.get("justification", "")
            lines.append(f"Step {num}: {claim} [{justification}]")

        return "\n".join(lines)
