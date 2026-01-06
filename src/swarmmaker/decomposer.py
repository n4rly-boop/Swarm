"""Problem decomposition agents."""


from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMClient
from .schemas import AgentCallMeta, DecompositionProposal, SwarmConfig


class Decomposer:
    """LLM wrapper that proposes recursive decompositions."""

    def __init__(self, llm: LLMClient, config: SwarmConfig) -> None:
        self.llm = llm
        self.config = config
        self.schema = DecompositionProposal.model_json_schema()

    def generate(self, problem: str, depth: int, step_id: int, batch_size: int) -> List[DecompositionProposal]:
        proposals: List[DecompositionProposal] = []
        for idx in range(batch_size):
            meta = AgentCallMeta(agent="decomposer", stage=f"decompose#{idx}", step_id=step_id, voter_id=idx)
            messages = [
                SystemMessage(
                    content=(
                        "You are a MAKER decomposition specialist.\n"
                        "Break the task into TWO smaller subproblems with a clear compose function.\n"
                        "If the task is already atomic, set is_atomic=true and explain briefly.\n"
                        "Rules:\n"
                        "- subproblem_a & subproblem_b must reduce the original task (no tautologies or restating givens)\n"
                        "- NEVER output 'solve y=... for y' or similar useless steps\n"
                        "- compose_fn must explain EXACTLY how to combine their answers without extra prose\n"
                        "- Keep each field concise; no chain-of-thought; respond with JSON only"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Depth: {depth}\n"
                        f"Problem: {problem}\n"
                        "Respond with DecompositionProposal JSON."
                    )
                ),
            ]
            result = self.llm.structured_completion(
                messages,
                meta=meta,
                model=self.config.model_decomposer,
                temperature=self.config.temperature_decomposer,
                schema_name="DecompositionProposal",
                schema=self.schema,
                parser=DecompositionProposal.model_validate,
            )
            proposals.append(result.content)
        return proposals
