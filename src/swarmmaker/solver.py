"""Atomic solver agents."""
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMClient
from .schemas import AgentCallMeta, AtomicSolution, SwarmConfig


class AtomicSolver:
    """Samples independent atomic solutions.

    From the MAKER paper:
    - Each agent receives minimal context (current state, strategy, prior move)
    - Temperature=0 for first vote, 0.1 for subsequent (configurable)
    - Outputs are validated and voted on
    """

    def __init__(self, llm: LLMClient, config: SwarmConfig) -> None:
        self.llm = llm
        self.config = config
        self.schema = AtomicSolution.model_json_schema()

    def solve(
        self,
        problem: str,
        step_id: int,
        batch_size: int,
        *,
        temperature: Optional[float] = None,
        round_idx: int = 0,
    ) -> List[AtomicSolution]:
        """Generate atomic solution candidates.

        Args:
            problem: Problem to solve.
            step_id: Current step ID.
            batch_size: Number of candidates to generate.
            temperature: Override temperature. If None, uses paper's strategy.
            round_idx: Current round index for temperature escalation.

        Returns:
            List of atomic solution candidates.
        """
        candidates: List[AtomicSolution] = []

        # Apply temperature escalation from paper:
        # - First vote: temperature = 0 (deterministic)
        # - Subsequent: temperature = 0.1 (slight diversity)
        if temperature is None:
            if round_idx == 0:
                temperature = self.config.thresholds.temperature_first_vote
            else:
                temperature = self.config.thresholds.temperature_subsequent

        for idx in range(batch_size):
            meta = AgentCallMeta(
                agent="solver",
                stage=f"solve#{idx}",
                step_id=step_id,
                voter_id=idx,
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an atomic problem solver.\n"
                        "Solve ONLY the problem provided. Show actual work in work_shown.\n"
                        "Limit work_shown to <=8 short lines and omit any meta commentary.\n\n"
                        "Required JSON schema:\n"
                        "{\n"
                        '  "solution": "your final answer (string, required)",\n'
                        '  "confidence": 0.0-1.0 (number, required),\n'
                        '  "work_shown": "intermediate steps (string, required)"\n'
                        "}\n\n"
                        "Output JSON only, no markdown fences or explanation."
                    )
                ),
                HumanMessage(
                    content=f"Problem: {problem}"
                ),
            ]
            result = self.llm.structured_completion(
                messages,
                meta=meta,
                model=self.config.get_model("execution"),
                temperature=temperature,
                schema_name="AtomicSolution",
                schema=self.schema,
                parser=AtomicSolution.model_validate,
                max_output_tokens=self.config.thresholds.max_output_tokens,
            )
            candidates.append(result.content)

        return candidates
