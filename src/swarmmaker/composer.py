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

    def compose(self, *, task: str, state: TaskState, step_id: int, feedback: Optional[str] = None) -> FinalAnswer:
        summary = self._state_summary(state)
        meta = AgentCallMeta(agent="composer", stage="finalize", step_id=step_id)
        prompts = [
            SystemMessage(
                content=(
                    "You are the FINAL COMPOSVER responsible for answering the original task.\n"
                    "Use only the provided structured summary of solved subproblems and notes.\n"
                    "Rules:\n"
                    "- Respond strictly with FinalAnswer JSON.\n"
                    "- Keep `answer` concise and directly address the task.\n"
                    "- If you include support, cap it to <=4 equations and <=4 points with numeric x/y values when relevant.\n"
                    "- work_shown or chain-of-thought must NOT appear anywhere.\n"
                    "- Confidence must reflect validation strength (0-1).\n"
                    "- Do not invent new subproblems; integrate what you have."
                )
            ),
            HumanMessage(
                content=(
                    f"Original task: {task.strip()}\n"
                    f"State summary (canonical JSON, trimmed):\n{summary}\n"
                    f"Verification feedback: {feedback or 'none'}\n"
                    "Return FinalAnswer JSON only."
                )
            ),
        ]
        result = self.llm.structured_completion(
            prompts,
            meta=meta,
            model=self.config.model_decomposer,
            temperature=self.config.temperature_composer,
            schema_name="FinalAnswer",
            schema=self.schema,
            parser=FinalAnswer.model_validate,
            max_output_tokens=400,
        )
        return result.content

    def _state_summary(self, state: TaskState) -> str:
        limit = self.config.progress_summary_limit
        facts = list(state.facts.items())
        trimmed_facts: List[Dict[str, Any]] = []
        for key, fact in facts[-limit:]:
            trimmed_facts.append(
                {
                    "key": key,
                    "problem": self._shorten(fact.problem),
                    "solution": self._shorten(fact.solution),
                    "confidence": fact.confidence,
                }
            )
        summary = {
            "draft_answer": self._shorten(state.draft_answer or "", max_chars=200),
            "notes": [self._shorten(note) for note in state.notes[-limit:]],
            "facts": trimmed_facts,
            "solved_subproblems": {k: self._shorten(v) for k, v in list(state.solved_subproblems.items())[-limit:]},
            "progress_version": state.progress_version,
        }
        serialized = canonical_json(summary)
        if len(serialized) > self.config.state_summary_char_limit:
            serialized = serialized[: self.config.state_summary_char_limit] + "..."
        return serialized

    def _shorten(self, text: str, *, max_chars: int = 160) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
