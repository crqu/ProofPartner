"""Lemma Breakdown — decomposes a proof goal into sub-lemmas.

Produces a LemmaTree with topologically ordered nodes and stable IDs.
Tags lemmas from prior published work for axiomatization.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    CRITIC_FEEDBACK_SECTION,
    LEMMA_BREAKDOWN_SYSTEM,
    LEMMA_BREAKDOWN_USER_TEMPLATE,
    PREAMBLE_CONTEXT_SECTION,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.proof import LemmaTree, NodeStatus, ProofNode

log = get_logger(__name__)

_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "is", "are",
     "that", "this", "it", "by", "with", "from", "as", "at", "be", "all", "any",
     "we", "show", "prove", "holds", "have", "let", "if", "then", "there", "exists"}
)

SIMILARITY_THRESHOLD = 0.7


def _normalize_tokens(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords."""
    import re as _re

    tokens = _re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _is_semantically_equivalent(a: str, b: str) -> bool:
    """Word-overlap similarity check (Jaccard) to detect circular axioms."""
    tokens_a = _normalize_tokens(a)
    tokens_b = _normalize_tokens(b)
    if not tokens_a or not tokens_b:
        return False
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) >= SIMILARITY_THRESHOLD


class LemmaBreakdown(BaseAgent):
    """Decomposes a theorem into topologically ordered sub-lemmas."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="lemma_breakdown", max_retries=2)
        self._llm = llm_client

    @staticmethod
    def format_critic_feedback(critic_issues: list[dict]) -> str:
        """Format critic issues into a string for the prompt."""
        lines = []
        for issue in critic_issues:
            issue_type = issue.get("issue_type", "unknown")
            node_id = issue.get("node_id", "unknown")
            description = issue.get("description", "")
            severity = issue.get("severity", "warning")
            suggested_fix = issue.get("suggested_fix", "")
            line = f"- [{severity}] {issue_type} at {node_id}: {description}"
            if suggested_fix:
                line += f"\n  Suggested fix: {suggested_fix}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_nl_proof_context(nl_ctx: dict) -> str:
        """Format an NLProofSketch dict into a prompt section."""
        lines = ["\n## Validated Informal Proof Sketch"]
        strategy = nl_ctx.get("overall_strategy", "")
        if strategy:
            lines.append(f"Strategy: {strategy}")
        for assumption in nl_ctx.get("assumptions", []):
            lines.append(f"- Assumption: {assumption}")
        for lemma in nl_ctx.get("key_lemmas", []):
            lines.append(f"- Key lemma: {lemma}")
        for i, step in enumerate(nl_ctx.get("proof_steps", []), 1):
            claim = step.get("claim", "")
            reasoning = step.get("reasoning", "")
            lines.append(f"\nStep {i}: {claim}")
            lines.append(f"  Reasoning: {reasoning}")
            for sc in step.get("sub_claims", []):
                lines.append(f"  - Sub-claim: {sc}")
        lines.append(
            "\nUse the above proof sketch to guide your decomposition. "
            "Align sub-lemmas with the proof steps above."
        )
        return "\n".join(lines)

    def _execute(self, context: AgentContext) -> AgentResult:
        statement_nl = context.task
        statement_lean = context.metadata.get("statement_lean", "")
        failed_attempts = context.metadata.get("failed_attempts", "None")
        parent_id = context.metadata.get("parent_id", "root")
        depth = context.metadata.get("depth", 0)
        critic_issues = context.metadata.get("critic_issues", [])
        lean_preamble = context.metadata.get("lean_preamble")
        nl_proof_context = context.metadata.get("nl_proof_context")
        tactic_hints = context.metadata.get("tactic_hints", "")

        user_content = LEMMA_BREAKDOWN_USER_TEMPLATE.format(
            statement_nl=statement_nl,
            statement_lean=statement_lean,
            failed_attempts=failed_attempts,
        )

        if nl_proof_context:
            user_content += self._format_nl_proof_context(nl_proof_context)

        if tactic_hints:
            user_content += (
                "\n\n## Tactic-Level Hints\n"
                f"{tactic_hints}\n\n"
                "Align your sub-lemma decomposition with these tactic suggestions."
            )

        if lean_preamble:
            user_content += PREAMBLE_CONTEXT_SECTION.format(
                lean_preamble=lean_preamble,
            )

        if critic_issues:
            issues_formatted = self.format_critic_feedback(critic_issues)
            user_content += CRITIC_FEEDBACK_SECTION.format(
                issues_formatted=issues_formatted,
            )

        response = self._llm.complete(
            system=LEMMA_BREAKDOWN_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        tree = self._parse_tree(response.content, parent_id, statement_nl, statement_lean, depth)

        log.info(
            "lemma_breakdown_done",
            node_count=len(tree.nodes),
            root=tree.root_id,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=tree.model_dump(),
            token_usage=response.token_usage,
        )

    def _parse_tree(
        self,
        response_text: str,
        parent_id: str,
        statement_nl: str,
        statement_lean: str,
        depth: int,
    ) -> LemmaTree:
        parsed = self._llm.extract_json(response_text)

        root_node = ProofNode(
            node_id=parent_id,
            statement_nl=statement_nl,
            statement_lean=statement_lean,
            depth=depth,
            status=NodeStatus.PENDING,
        )

        nodes: dict[str, ProofNode] = {parent_id: root_node}
        child_ids: list[str] = []

        if isinstance(parsed, dict):
            for item in parsed.get("lemmas", []):
                node_id = item.get("node_id", f"lemma_{len(nodes)}")
                is_prior_work = item.get("from_prior_work", False)
                child_statement_nl = item.get("statement_nl", "")

                if is_prior_work and _is_semantically_equivalent(
                    child_statement_nl, statement_nl
                ):
                    log.warning(
                        "circular_axiom_guard",
                        node_id=node_id,
                        child_statement=child_statement_nl,
                        root_statement=statement_nl,
                    )
                    is_prior_work = False

                child_node = ProofNode(
                    node_id=node_id,
                    statement_nl=child_statement_nl,
                    depth=depth + 1,
                    parent_id=parent_id,
                    status=NodeStatus.PENDING,
                    from_prior_work=is_prior_work,
                    source_reference=item.get("source_reference") if is_prior_work else None,
                )
                item.get("depends_on", [])
                child_node.children = []
                nodes[node_id] = child_node
                child_ids.append(node_id)

            root_node.children = child_ids
            topo_order = parsed.get("topological_order", child_ids)
        else:
            topo_order = []

        valid_topo = [nid for nid in topo_order if nid in nodes]
        remaining = [nid for nid in nodes if nid not in valid_topo]
        valid_topo.extend(remaining)

        return LemmaTree(
            root_id=parent_id,
            nodes=nodes,
            topological_order=valid_topo,
        )
