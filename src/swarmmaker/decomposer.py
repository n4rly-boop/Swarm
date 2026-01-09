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
                        "You are a MAKER decomposition specialist.\n\n"
                        "CRITICAL: Mark is_atomic=true if ANY of these apply:\n"
                        "- Single arithmetic operation (e.g., '5+7', 'x+5=12')\n"
                        "- Simple algebraic step (e.g., 'isolate x', 'substitute y=3')\n"
                        "- One-step calculation or lookup\n"
                        "- Problem can be solved in one mental step\n"
                        "- Problem is about identifying/extracting something\n\n"
                        "Mark is_atomic=false ONLY if the problem has MULTIPLE DISTINCT steps that:\n"
                        "- Require solving separate sub-tasks\n"
                        "- Then combining their results\n\n"
                        "Examples of ATOMIC (is_atomic=true):\n"
                        "- 'What is 5 + 7?' → atomic\n"
                        "- 'Solve x + 5 = 12' → atomic (one step: x = 12 - 5)\n"
                        "- 'Substitute x=3 into y=2x' → atomic\n"
                        "- 'Isolate x in x + y = 7' → atomic\n\n"
                        "Examples of NON-ATOMIC (is_atomic=false):\n"
                        "- 'Solve system: x+y=7, 2x-y=2, then calculate x*y' → 2 parts: solve system + multiply\n"
                        "- 'Find intersection points and sum their coordinates' → find + sum\n\n"
                        "Required JSON schema:\n"
                        "{\n"
                        '  "subproblem_a": "first subproblem (string)",\n'
                        '  "subproblem_b": "second subproblem (string)",\n'
                        '  "compose_fn": "how to combine (string)",\n'
                        '  "is_atomic": true/false,\n'
                        '  "rationale": "why (string)"\n'
                        "}\n\n"
                        "If is_atomic=true: set subproblem_a to the problem, subproblem_b='N/A', compose_fn='Return directly'."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Depth: {depth}\n"
                        f"Problem: {problem}\n"
                        "Output JSON only."
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
