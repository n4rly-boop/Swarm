"""Completeness checker agent for verifying task requirements are addressed."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import AgentCallMeta, CompletenessResult, SwarmConfig, TaskState

if TYPE_CHECKING:
    from .llm import LLMClient


class CompletenessChecker:
    """LLM agent that verifies if all task requirements are addressed in the answer."""

    def __init__(self, llm: "LLMClient", config: SwarmConfig) -> None:
        self.llm = llm
        self.config = config
        self.schema = CompletenessResult.model_json_schema()

    def check(
        self,
        task: str,
        answer: str,
        state: TaskState,
        step_id: int,
    ) -> CompletenessResult:
        """Check if the answer addresses all requirements in the task.

        Args:
            task: The original task description.
            answer: The proposed answer to verify.
            state: Current task state with solved subproblems.
            step_id: Current step ID for logging.

        Returns:
            CompletenessResult with requirements status and any missing work.
        """
        meta = AgentCallMeta(agent="completeness", stage="verify", step_id=step_id)

        # Include context from solved subproblems if available
        context = ""
        if state.solved_subproblems:
            solved = list(state.solved_subproblems.items())[-3:]  # Last 3
            context = "\n".join(f"- {k}: {v}" for k, v in solved)
            context = f"\n\nPrior solved work:\n{context}"

        messages = [
            SystemMessage(
                content=(
                    "You are a completeness verifier for a task-solving system.\n"
                    "Given a task and an answer, determine if ALL requirements are fully addressed.\n\n"
                    "Rules:\n"
                    "- Extract each distinct requirement from the task\n"
                    "- For each requirement, determine if it is ADDRESSED or MISSING\n"
                    "- Be strict: vague or partial answers count as MISSING\n"
                    "- If something is MISSING, provide a specific actionable task in missing_work\n"
                    "- The missing_work items should be concrete problems that can be solved\n"
                    "- Respond with JSON only, no prose"
                )
            ),
            HumanMessage(
                content=(
                    f"Task: {task}\n\n"
                    f"Answer: {answer}"
                    f"{context}\n\n"
                    "Analyze completeness and return CompletenessResult JSON."
                )
            ),
        ]

        result = self.llm.structured_completion(
            messages,
            meta=meta,
            model=self.config.model_solver,
            temperature=0.1,  # Low temperature for consistent verification
            schema_name="CompletenessResult",
            schema=self.schema,
            parser=CompletenessResult.model_validate,
        )
        return result.content
