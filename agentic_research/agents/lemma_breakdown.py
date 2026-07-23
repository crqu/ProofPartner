"""Lemma Breakdown — decomposes a proof goal into sub-lemmas.

Produces a LemmaTree with topologically ordered nodes and stable IDs.
Tags lemmas from prior published work for axiomatization.
Supports best-of-k parallel decomposition with MVP scoring.
"""

from __future__ import annotations

import asyncio

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
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, NodeStatus, ProofNode

log = get_logger(__name__)

DEFAULT_DECOMPOSITION_K = 3

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

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        decomposition_k: int = 1,
    ) -> None:
        super().__init__(name="lemma_breakdown", max_retries=2)
        self._llm = llm_client
        self._decomposition_k = decomposition_k

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

    @staticmethod
    def _score_decomposition(tree: LemmaTree) -> float:
        """Score a decomposition tree using brevity and structural balance.

        Returns a float in [0, 1] — higher is better.
        """
        node_count = len(tree.nodes)
        brevity = 1.0 / (1.0 + node_count / 8.0)

        if node_count <= 1:
            return brevity

        depths = [n.depth for n in tree.nodes.values()]
        max_depth = max(depths) if depths else 1
        if max_depth == 0:
            return brevity

        mean_depth = sum(depths) / len(depths)
        depth_variance = sum((d - mean_depth) ** 2 for d in depths) / len(depths)
        balance = 1.0 - min(depth_variance / max_depth, 1.0)

        return 0.5 * brevity + 0.5 * balance

    def _build_user_content(self, context: AgentContext) -> str:
        """Build the prompt user content from context metadata."""
        statement_nl = context.task
        statement_lean = context.metadata.get("statement_lean", "")
        failed_attempts = context.metadata.get("failed_attempts", "None")
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

        return user_content

    def _generate_single_candidate(
        self,
        user_content: str,
        temperature: float,
        parent_id: str,
        statement_nl: str,
        statement_lean: str,
        depth: int,
    ) -> tuple[LemmaTree, TokenUsage]:
        """Generate a single decomposition candidate at a given temperature."""
        response = self._llm.complete(
            system=LEMMA_BREAKDOWN_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature,
            use_cache=(temperature <= 0.3),
        )
        tree = self._parse_tree(
            response.content, parent_id, statement_nl, statement_lean, depth,
        )
        score = self._score_decomposition(tree)
        tree.decomposition_score = score
        return tree, response.token_usage

    def _execute(self, context: AgentContext) -> AgentResult:
        statement_nl = context.task
        statement_lean = context.metadata.get("statement_lean", "")
        parent_id = context.metadata.get("parent_id", "root")
        depth = context.metadata.get("depth", 0)
        k = self._decomposition_k

        user_content = self._build_user_content(context)
        temperatures = [0.3 + i * 0.05 for i in range(k)]

        if k <= 1:
            tree, token_usage = self._generate_single_candidate(
                user_content, temperatures[0],
                parent_id, statement_nl, statement_lean, depth,
            )
            log.info(
                "lemma_breakdown_done",
                node_count=len(tree.nodes),
                root=tree.root_id,
                score=tree.decomposition_score,
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                result=tree.model_dump(),
                token_usage=token_usage,
            )

        candidates = self._run_parallel_candidates(
            user_content, temperatures,
            parent_id, statement_nl, statement_lean, depth,
        )

        total_tokens = TokenUsage()
        best_tree: LemmaTree | None = None
        best_score = -1.0
        for tree, usage in candidates:
            total_tokens.input_tokens += usage.input_tokens
            total_tokens.output_tokens += usage.output_tokens
            total_tokens.cache_creation_input_tokens += usage.cache_creation_input_tokens
            total_tokens.cache_read_input_tokens += usage.cache_read_input_tokens
            score = tree.decomposition_score or 0.0
            if score > best_score:
                best_score = score
                best_tree = tree

        assert best_tree is not None

        log.info(
            "lemma_breakdown_best_of_k",
            k=k,
            best_score=best_score,
            node_count=len(best_tree.nodes),
            root=best_tree.root_id,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            result=best_tree.model_dump(),
            token_usage=total_tokens,
        )

    def _run_parallel_candidates(
        self,
        user_content: str,
        temperatures: list[float],
        parent_id: str,
        statement_nl: str,
        statement_lean: str,
        depth: int,
    ) -> list[tuple[LemmaTree, TokenUsage]]:
        """Run k decomposition candidates in parallel using asyncio."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return self._run_sequential_candidates(
                user_content, temperatures,
                parent_id, statement_nl, statement_lean, depth,
            )

        return asyncio.run(
            self._run_async_candidates(
                user_content, temperatures,
                parent_id, statement_nl, statement_lean, depth,
            )
        )

    async def _run_async_candidates(
        self,
        user_content: str,
        temperatures: list[float],
        parent_id: str,
        statement_nl: str,
        statement_lean: str,
        depth: int,
    ) -> list[tuple[LemmaTree, TokenUsage]]:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                self._generate_single_candidate,
                user_content, temp,
                parent_id, statement_nl, statement_lean, depth,
            )
            for temp in temperatures
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[tuple[LemmaTree, TokenUsage]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                log.warning(
                    "lemma_breakdown_candidate_error",
                    candidate_index=i,
                    error=str(result),
                )
            else:
                candidates.append(result)

        if not candidates:
            tree, usage = self._generate_single_candidate(
                user_content, temperatures[0],
                parent_id, statement_nl, statement_lean, depth,
            )
            candidates.append((tree, usage))

        return candidates

    def _run_sequential_candidates(
        self,
        user_content: str,
        temperatures: list[float],
        parent_id: str,
        statement_nl: str,
        statement_lean: str,
        depth: int,
    ) -> list[tuple[LemmaTree, TokenUsage]]:
        """Fallback when already inside an event loop."""
        candidates: list[tuple[LemmaTree, TokenUsage]] = []
        for i, temp in enumerate(temperatures):
            try:
                result = self._generate_single_candidate(
                    user_content, temp,
                    parent_id, statement_nl, statement_lean, depth,
                )
                candidates.append(result)
            except Exception as exc:
                log.warning(
                    "lemma_breakdown_candidate_error",
                    candidate_index=i,
                    error=str(exc),
                )
        if not candidates:
            tree, usage = self._generate_single_candidate(
                user_content, temperatures[0],
                parent_id, statement_nl, statement_lean, depth,
            )
            candidates.append((tree, usage))
        return candidates

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
