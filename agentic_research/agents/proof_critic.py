"""Proof Critic — pre-formalization soundness audit for lemma trees.

Uses a propose-then-confirm loop: first identifies candidate issues,
then attempts to refute each one adversarially. Only confirmed issues
survive to the output.
"""

from __future__ import annotations

import json

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    PROOF_CRITIC_CONFIRM_TEMPLATE,
    PROOF_CRITIC_SYSTEM,
    PROOF_CRITIC_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.proof import (
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueResult,
    LemmaTree,
)

log = get_logger(__name__)


class ProofCritic(BaseAgent):
    """Audits a LemmaTree for logical soundness before Lean translation."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="proof_critic", max_retries=1)
        self._llm = llm_client

    def _execute(self, context: AgentContext) -> AgentResult:
        tree_data = context.metadata.get("lemma_tree")
        if not tree_data:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                error_message="No lemma_tree in context metadata",
            )

        tree = LemmaTree.model_validate(tree_data)
        statement_nl = context.task
        statement_lean = context.metadata.get("statement_lean", "")

        critique = self.critique(
            tree=tree,
            statement_nl=statement_nl,
            statement_lean=statement_lean,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=critique.model_dump(),
            token_usage=critique.token_usage,
        )

    def critique(
        self,
        *,
        tree: LemmaTree,
        statement_nl: str,
        statement_lean: str = "",
    ) -> CritiqueResult:
        """Run the propose-then-confirm critique loop."""
        total_tokens = TokenUsage()
        tree_desc = self._describe_tree(tree)

        proposed, propose_tokens = self._propose_issues(
            statement_nl=statement_nl,
            statement_lean=statement_lean,
            tree_description=tree_desc,
        )
        total_tokens.input_tokens += propose_tokens.input_tokens
        total_tokens.output_tokens += propose_tokens.output_tokens

        if not proposed:
            log.info("proof_critic_no_issues")
            return CritiqueResult(passed=True, token_usage=total_tokens)

        confirmed, confirm_tokens = self._confirm_issues(
            statement_nl=statement_nl,
            proposed_issues=proposed,
            tree_description=tree_desc,
        )
        total_tokens.input_tokens += confirm_tokens.input_tokens
        total_tokens.output_tokens += confirm_tokens.output_tokens

        has_blocking = any(i.severity == "blocking" for i in confirmed)

        log.info(
            "proof_critic_done",
            proposed=len(proposed),
            confirmed=len(confirmed),
            blocking=has_blocking,
        )

        return CritiqueResult(
            issues=confirmed,
            passed=not has_blocking,
            token_usage=total_tokens,
        )

    def _propose_issues(
        self,
        statement_nl: str,
        statement_lean: str,
        tree_description: str,
    ) -> tuple[list[CritiqueIssue], TokenUsage]:
        user_content = PROOF_CRITIC_USER_TEMPLATE.format(
            statement_nl=statement_nl,
            statement_lean=statement_lean or "-- not yet formalized",
            lemma_tree_description=tree_description,
        )

        response = self._llm.complete(
            system=PROOF_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.3,
            use_cache=True,
        )

        issues = self._parse_issues(response.content)
        return issues, response.token_usage

    def _confirm_issues(
        self,
        statement_nl: str,
        proposed_issues: list[CritiqueIssue],
        tree_description: str,
    ) -> tuple[list[CritiqueIssue], TokenUsage]:
        issues_json = json.dumps(
            [i.model_dump() for i in proposed_issues], indent=2
        )

        user_content = PROOF_CRITIC_CONFIRM_TEMPLATE.format(
            statement_nl=statement_nl,
            proposed_issues=issues_json,
            lemma_tree_description=tree_description,
        )

        response = self._llm.complete(
            system=PROOF_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )

        confirmed = self._parse_confirmed_issues(response.content)
        return confirmed, response.token_usage

    def _parse_issues(self, response_text: str) -> list[CritiqueIssue]:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            return []

        issues = []
        for item in parsed.get("issues", []):
            issue = self._make_issue(item)
            if issue:
                issues.append(issue)
        return issues

    def _parse_confirmed_issues(self, response_text: str) -> list[CritiqueIssue]:
        parsed = self._llm.extract_json(response_text)
        if not isinstance(parsed, dict):
            return []

        issues = []
        for item in parsed.get("confirmed_issues", []):
            issue = self._make_issue(item, confirmed=True)
            if issue:
                issues.append(issue)
        return issues

    def _make_issue(
        self, item: dict, confirmed: bool = False
    ) -> CritiqueIssue | None:
        raw_type = item.get("issue_type", "")
        try:
            issue_type = CritiqueIssueType(raw_type)
        except ValueError:
            return None

        return CritiqueIssue(
            issue_type=issue_type,
            node_id=item.get("node_id", "unknown"),
            description=item.get("description", ""),
            severity=item.get("severity", "warning"),
            suggested_fix=item.get("suggested_fix", ""),
            confirmed=confirmed,
        )

    def _describe_tree(self, tree: LemmaTree) -> str:
        parts = []
        for node_id in tree.topological_order:
            node = tree.get_node(node_id)
            if not node:
                continue
            parent_info = f" (child of {node.parent_id})" if node.parent_id else " (ROOT)"
            prior = " [FROM PRIOR WORK]" if node.from_prior_work else ""
            parts.append(
                f"- {node.node_id}{parent_info}{prior}: {node.statement_nl}"
            )
        return "\n".join(parts)
