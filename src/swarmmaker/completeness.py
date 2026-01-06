"""Completeness checker agent for verifying task requirements are addressed."""
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
                    "- Extract ONLY requirements EXPLICITLY stated in the task text\n"
                    "- Do NOT invent requirements like 'show work' or 'provide steps' unless explicitly asked\n"
                    "- For each requirement, determine if it is ADDRESSED or MISSING\n"
                    "- Be strict: vague or partial answers count as MISSING\n\n"
                    "For missing_work items (CRITICAL):\n"
                    "- Workers have ZERO context - they cannot see the task or answer\n"
                    "- Each missing_work MUST be a self-contained calculation with ALL values inline\n"
                    "- Use actual numbers from the answer, formatted as a direct computation\n"
                    "Examples:\n"
                    "  BAD: 'Calculate the sum of vectors to the points'\n"
                    "  BAD: 'Calculate vector from origin to (3,4)'\n"
                    "  GOOD: 'What is (3,4) + (1,-2)?'\n"
                    "  GOOD: 'Compute 3 + 1 and 4 + (-2)'\n"
                    "- Respond with JSON only, no prose"
                )
            ),
            HumanMessage(
                content=(
                    f"Task: {task}\n\n"
                    f"Answer: {answer}"
                    f"{context}\n\n"
                    "Extract ONLY explicit requirements. Write missing_work as direct computations with all values."
                )
            ),
        ]

        result = self.llm.structured_completion(
            messages,
            meta=meta,
            model=self.config.model_decomposer,
            temperature=0.1,  # Low temperature for consistent verification
            schema_name="CompletenessResult",
            schema=self.schema,
            parser=CompletenessResult.model_validate,
        )
        return result.content
