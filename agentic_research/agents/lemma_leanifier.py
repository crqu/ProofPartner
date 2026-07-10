"""Lemma Leanifier — translates sub-lemmas to Lean 4 statements.

Each sub-lemma is translated into a Lean 4 theorem statement with
body = sorry.  The statement is validated to compile before proceeding.
"""

from __future__ import annotations

import re

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    AXIOM_LEANIFY_SYSTEM,
    AXIOM_LEANIFY_USER_TEMPLATE,
    LEMMA_LEANIFY_FEEDBACK_TEMPLATE,
    LEMMA_LEANIFY_SYSTEM,
    LEMMA_LEANIFY_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
    TokenUsage,
)
from agentic_research.models.proof import LemmaTree, ProofNode
from agentic_research.models.tools import CompilationStatus
from agentic_research.tools.lean_repl import LeanRepl

log = get_logger(__name__)

MAX_COMPILE_RETRIES = 3


def _extract_lean_code(text: str) -> str:
    match = re.search(r"```lean\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class LemmaLeanifier(BaseAgent):
    """Translates sub-lemmas to compilable Lean 4 sorry-statements."""

    def __init__(
        self,
        llm_client: LLMClient,
        lean_repl: LeanRepl,
        *,
        max_compile_retries: int = MAX_COMPILE_RETRIES,
        lean_preamble: str | None = None,
    ) -> None:
        super().__init__(name="lemma_leanifier", max_retries=1)
        self._llm = llm_client
        self._repl = lean_repl
        self._max_compile_retries = max_compile_retries
        self._lean_preamble = lean_preamble

    def _compile(self, lean_code: str):
        """Execute lean_code in the REPL, prepending the preamble if set."""
        if self._lean_preamble:
            full_code = self._lean_preamble + "\n\n" + lean_code
        else:
            full_code = lean_code
        return self._repl.execute(full_code)

    def _definitions_context(self) -> str:
        """Return an LLM prompt section describing available definitions."""
        if not self._lean_preamble:
            return ""
        return (
            "\n\n## Available Definitions\n"
            "The following Lean 4 definitions are already in scope:\n"
            + self._lean_preamble
        )

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
        failed_nodes: list[str] = []

        for node_id in tree.topological_order:
            node = tree.get_node(node_id)
            if not node or node.statement_lean:
                continue
            if node.node_id == tree.root_id and node.statement_lean:
                continue

            parent = tree.get_node(node.parent_id) if node.parent_id else None
            parent_stmt = parent.statement_lean if parent else ""
            siblings = self._get_sibling_statements(tree, node)

            lean_code, tokens = self._leanify_node(node, parent_stmt, siblings)
            total_tokens.input_tokens += tokens.input_tokens
            total_tokens.output_tokens += tokens.output_tokens

            if lean_code:
                node.statement_lean = lean_code
                log.info("lemma_leanified", node_id=node_id)
            else:
                failed_nodes.append(node_id)
                log.warning("lemma_leanify_failed", node_id=node_id)

        status = AgentStatus.SUCCESS if not failed_nodes else AgentStatus.FAILURE
        return AgentResult(
            agent_name=self.name,
            status=status,
            result=tree.model_dump(),
            token_usage=total_tokens,
            error_message=f"Failed to leanify: {failed_nodes}" if failed_nodes else None,
        )

    def _leanify_node(
        self, node: ProofNode, parent_statement: str, sibling_statements: str
    ) -> tuple[str | None, TokenUsage]:
        if node.from_prior_work:
            axiom_code, axiom_tokens = self._axiomatize_node(
                node, parent_statement, sibling_statements
            )
            if axiom_code:
                return axiom_code, axiom_tokens
            log.info("axiom_leanify_fallback", node_id=node.node_id)

        total_tokens = TokenUsage()

        user_content = LEMMA_LEANIFY_USER_TEMPLATE.format(
            node_id=node.node_id,
            statement_nl=node.statement_nl,
            parent_statement=parent_statement or "-- (root theorem)",
            sibling_statements=sibling_statements or "-- none",
        )

        if node.proof_sketch_nl:
            user_content += (
                f"\n\n## Proof Sketch\n{node.proof_sketch_nl}\n"
                "Use this sketch to inform the Lean statement structure."
            )

        user_content += self._definitions_context()

        response = self._llm.complete(
            system=LEMMA_LEANIFY_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )
        total_tokens.input_tokens += response.token_usage.input_tokens
        total_tokens.output_tokens += response.token_usage.output_tokens

        lean_code = _extract_lean_code(response.content)
        compilation = self._compile(lean_code)

        if compilation.compilation_status == CompilationStatus.OK:
            return lean_code, total_tokens

        for retry in range(1, self._max_compile_retries + 1):
            log.info("lemma_leanify_retry", node_id=node.node_id, retry=retry)
            errors = "\n".join(compilation.errors) if compilation.errors else "Unknown error"

            feedback_content = LEMMA_LEANIFY_FEEDBACK_TEMPLATE.format(
                node_id=node.node_id,
                statement_nl=node.statement_nl,
                previous_attempt=lean_code,
                errors=errors,
            )

            response = self._llm.complete(
                system=LEMMA_LEANIFY_SYSTEM,
                messages=[{"role": "user", "content": feedback_content}],
                temperature=0.0,
                use_cache=True,
            )
            total_tokens.input_tokens += response.token_usage.input_tokens
            total_tokens.output_tokens += response.token_usage.output_tokens

            lean_code = _extract_lean_code(response.content)
            compilation = self._compile(lean_code)

            if compilation.compilation_status == CompilationStatus.OK:
                return lean_code, total_tokens

        return None, total_tokens

    def _axiomatize_node(
        self, node: ProofNode, parent_statement: str, sibling_statements: str
    ) -> tuple[str | None, TokenUsage]:
        """Produce a Lean 4 axiom declaration for a from_prior_work node."""
        total_tokens = TokenUsage()

        user_content = AXIOM_LEANIFY_USER_TEMPLATE.format(
            node_id=node.node_id,
            statement_nl=node.statement_nl,
            source_reference=node.source_reference or "unspecified prior work",
            parent_statement=parent_statement or "-- (root theorem)",
            sibling_statements=sibling_statements or "-- none",
        )

        user_content += self._definitions_context()

        response = self._llm.complete(
            system=AXIOM_LEANIFY_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            use_cache=True,
        )
        total_tokens.input_tokens += response.token_usage.input_tokens
        total_tokens.output_tokens += response.token_usage.output_tokens

        lean_code = _extract_lean_code(response.content)
        compilation = self._compile(lean_code)

        if compilation.compilation_status == CompilationStatus.OK:
            log.info("axiom_leanified", node_id=node.node_id)
            return lean_code, total_tokens

        return None, total_tokens

    def _get_sibling_statements(self, tree: LemmaTree, node: ProofNode) -> str:
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
