"""Final composer that integrates solved subproblems into a single answer."""
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMClient
from .schemas import AgentCallMeta, FinalAnswer, SwarmConfig, TaskState, canonical_json


class FinalComposer:
    """LLM helper that produces the final answer payload."""

    def __init__(self, llm: LLMClient, config: SwarmConfig) -> None:
        self.llm = llm
        self.config = config
        self.schema = FinalAnswer.model_json_schema()

    def compose(
        self,
        *,
        task: str,
        state: TaskState,
        step_id: int,
        feedback: Optional[str] = None,
    ) -> FinalAnswer:
        """Compose the final answer from task state.

        Args:
            task: Original task description.
            state: Current task state with solved subproblems.
            step_id: Current step ID for logging.
            feedback: Optional verification feedback from previous attempt.

        Returns:
            FinalAnswer payload.
        """
        summary = self._state_summary(state)
        meta = AgentCallMeta(agent="composer", stage="finalize", step_id=step_id)

        prompts = [
            SystemMessage(
                content=(
                    "You are the FINAL COMPOSER responsible for answering the original task.\n"
                    "Use only the provided structured summary of solved subproblems and notes.\n"
                    "Rules:\n"
                    "- Keep `answer` concise and directly address the task.\n"
                    "- Confidence must reflect validation strength (0-1).\n"
                    "- Do not invent new subproblems; integrate what you have.\n\n"
                    "Required JSON schema:\n"
                    "{\n"
                    '  "answer": "direct response to the task (string, required)",\n'
                    '  "confidence": 0.0-1.0 (number, required),\n'
                    '  "support": {"summary": "", "equations": [], "points": []} (optional)\n'
                    "}\n\n"
                    "Output JSON only, no markdown fences."
                )
            ),
            HumanMessage(
                content=(
                    f"Original task: {task.strip()}\n"
                    f"State summary:\n{summary}\n"
                    f"Verification feedback: {feedback or 'none'}"
                )
            ),
        ]

        result = self.llm.structured_completion(
            prompts,
            meta=meta,
            model=self.config.get_model("reasoning"),
            temperature=self.config.thresholds.temperature_first_vote,  # Low temp for composition
            schema_name="FinalAnswer",
            schema=self.schema,
            parser=FinalAnswer.model_validate,
            max_output_tokens=400,
        )
        return result.content

    def _state_summary(self, state: TaskState) -> str:
        """Create a summary of task state for the composer prompt.

        Args:
            state: Current task state.

        Returns:
            JSON-formatted summary string.
        """
        limit = self.config.thresholds.max_facts
        char_limit = self.config.thresholds.state_summary_char_limit

        facts = list(state.facts.items())
        trimmed_facts: List[Dict[str, Any]] = []

        for key, fact in facts[-limit:]:
            trimmed_facts.append({
                "key": key,
                "problem": self._shorten(fact.problem),
                "solution": self._shorten(fact.solution),
                "confidence": fact.confidence,
            })

        summary = {
            "draft_answer": self._shorten(state.draft_answer or "", max_chars=200),
            "notes": [self._shorten(note) for note in state.notes[-limit:]],
            "facts": trimmed_facts,
            "solved_subproblems": {
                k: self._shorten(v)
                for k, v in list(state.solved_subproblems.items())[-limit:]
            },
            "progress_version": state.progress_version,
        }

        serialized = canonical_json(summary)
        if len(serialized) > char_limit:
            serialized = serialized[:char_limit] + "..."

        return serialized

    def _shorten(self, text: str, *, max_chars: int = 160) -> str:
        """Shorten text to max_chars with ellipsis."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
