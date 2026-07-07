"""Lemma Breakdown — decomposes a proof goal into sub-lemmas.

Produces a LemmaTree with topologically ordered nodes and stable IDs.
Tags lemmas from prior published work for axiomatization.
"""

from __future__ import annotations

from agentic_research.agents.base import BaseAgent
from agentic_research.agents.llm_client import LLMClient
from agentic_research.agents.prompt_templates import (
    LEMMA_BREAKDOWN_SYSTEM,
    LEMMA_BREAKDOWN_USER_TEMPLATE,
)
from agentic_research.logging import get_logger
from agentic_research.models.agents import (
    AgentContext,
    AgentResult,
    AgentStatus,
)
from agentic_research.models.proof import LemmaTree, NodeStatus, ProofNode

log = get_logger(__name__)


class LemmaBreakdown(BaseAgent):
    """Decomposes a theorem into topologically ordered sub-lemmas."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="lemma_breakdown", max_retries=2)
        self._llm = llm_client

    def _execute(self, context: AgentContext) -> AgentResult:
        statement_nl = context.task
        statement_lean = context.metadata.get("statement_lean", "")
        failed_attempts = context.metadata.get("failed_attempts", "None")
        parent_id = context.metadata.get("parent_id", "root")
        depth = context.metadata.get("depth", 0)

        user_content = LEMMA_BREAKDOWN_USER_TEMPLATE.format(
            statement_nl=statement_nl,
            statement_lean=statement_lean,
            failed_attempts=failed_attempts,
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
                child_node = ProofNode(
                    node_id=node_id,
                    statement_nl=item.get("statement_nl", ""),
                    depth=depth + 1,
                    parent_id=parent_id,
                    status=NodeStatus.PENDING,
                    from_prior_work=item.get("from_prior_work", False),
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
