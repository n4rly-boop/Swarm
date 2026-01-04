"""Atomic solver agents."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMClient
from .schemas import AgentCallMeta, AtomicSolution, SwarmConfig


class AtomicSolver:
    """Samples independent atomic solutions."""

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
    ) -> List[AtomicSolution]:
        candidates: List[AtomicSolution] = []
        for idx in range(batch_size):
            meta = AgentCallMeta(agent="solver", stage=f"solve#{idx}", step_id=step_id, voter_id=idx)
            messages = [
                SystemMessage(
                    content=(
                        "You are an atomic problem solver.\n"
                        "Solve ONLY the problem provided. Show actual work in work_shown.\n"
                        "Limit work_shown to <=8 short lines of math and omit any meta commentary.\n"
                        "Respond strictly with AtomicSolution JSON."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Problem: {problem}\n"
                        "Return fields: solution (string answer), confidence (0-1), work_shown (steps)."
                    )
                ),
            ]
            result = self.llm.structured_completion(
                messages,
                meta=meta,
                model=self.config.model_solver,
                temperature=temperature or self.config.temperature_solver,
                schema_name="AtomicSolution",
                schema=self.schema,
                parser=AtomicSolution.model_validate,
            )
            candidates.append(result.content)
        return candidates
