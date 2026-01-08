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

    def generate(
        self,
        problem: str,
        depth: int,
        step_id: int,
        batch_size: int,
        *,
        temperature: float | None = None,
    ) -> List[DecompositionProposal]:
        """Generate decomposition proposals.

        Args:
            problem: Problem to decompose.
            depth: Current recursion depth.
            step_id: Current step ID.
            batch_size: Number of proposals to generate.
            temperature: Override temperature (uses config default if None).

        Returns:
            List of decomposition proposals.
        """
        proposals: List[DecompositionProposal] = []

        # Use temperature escalation from config thresholds
        default_temp = self.config.thresholds.temperature_subsequent

        for idx in range(batch_size):
            meta = AgentCallMeta(
                agent="decomposer",
                stage=f"decompose#{idx}",
                step_id=step_id,
                voter_id=idx,
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a MAKER decomposition specialist.\n"
                        "Break the task into TWO smaller subproblems with a clear compose function.\n"
                        "If the task is already atomic (cannot be meaningfully broken down), set is_atomic=true.\n\n"
                        "Rules:\n"
                        "- subproblem_a & subproblem_b must reduce the original task (no tautologies)\n"
                        "- NEVER output 'solve y=... for y' or similar useless steps\n"
                        "- compose_fn must explain EXACTLY how to combine their answers\n"
                        "- Keep each field concise; respond with JSON only\n\n"
                        "Required JSON schema:\n"
                        "{\n"
                        '  "subproblem_a": "first subproblem (string, required)",\n'
                        '  "subproblem_b": "second subproblem (string, required)",\n'
                        '  "compose_fn": "how to combine results (string, required)",\n'
                        '  "is_atomic": true/false (boolean, required),\n'
                        '  "rationale": "why this decomposition (string, required)"\n'
                        "}\n\n"
                        "If is_atomic=true, still provide subproblem_a (the original problem), "
                        "subproblem_b (empty or 'N/A'), and compose_fn ('Return result directly')."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Depth: {depth}\n"
                        f"Problem: {problem}\n"
                        "Output JSON only, no markdown fences."
                    )
                ),
            ]
            result = self.llm.structured_completion(
                messages,
                meta=meta,
                model=self.config.get_model("reasoning"),
                temperature=temperature if temperature is not None else default_temp,
                schema_name="DecompositionProposal",
                schema=self.schema,
                parser=DecompositionProposal.model_validate,
            )
            proposals.append(result.content)

        return proposals
