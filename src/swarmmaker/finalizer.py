"""Best-effort finalization for SwarmMaker.

This module guarantees that a final answer is always produced, even when
the normal orchestration flow fails or is interrupted.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from .domain_state import CodeState, CreativeState, MathState
from .schemas import AgentCallMeta

if TYPE_CHECKING:
    from .graph import GraphState, RuntimeContext
    from .llm import LLMClient
    from .schemas import SwarmConfig


@dataclass
class FinalizationResult:
    """Result of best-effort finalization.

    Attributes:
        final_answer: The synthesized final answer
        confidence: Confidence in answer quality (0-1)
        synthesis_method: How the answer was generated
        is_partial: Whether this is a partial/incomplete answer
    """

    final_answer: str
    confidence: float
    synthesis_method: str
    is_partial: bool


class BestEffortFinalizer:
    """Synthesizes final answers when normal flow fails.

    Tries multiple strategies in order of preference to guarantee
    that some output is always produced.
    """

    def __init__(self, llm_client: "LLMClient", config: "SwarmConfig"):
        """Initialize finalizer.

        Args:
            llm_client: LLM client for synthesis
            config: Swarm configuration
        """
        self.llm = llm_client
        self.config = config

    def finalize(
        self, task: str, state: "GraphState", runtime: "RuntimeContext"
    ) -> FinalizationResult:
        """Produce best-effort final answer from partial progress.

        Args:
            task: Original task description
            state: Current graph state
            runtime: Runtime context

        Returns:
            FinalizationResult with synthesized answer
        """
        # STRATEGY 1: Try LLM synthesis from domain state + progress
        if runtime.metrics.tokens_total() < self.config.max_total_tokens * 0.95:
            try:
                return self._llm_synthesis(task, state, runtime)
            except Exception as e:
                runtime.display.log_event(f"LLM synthesis failed: {e}")

        # STRATEGY 2: Use last draft if exists
        draft = state.get("draft_answer")
        if draft and len(draft.strip()) > 20:
            return FinalizationResult(
                final_answer=draft,
                confidence=0.5,
                synthesis_method="last_draft",
                is_partial=True,
            )

        # STRATEGY 3: Synthesize from domain state structured fields
        domain_state = state.get("domain_state")
        if domain_state:
            try:
                return self._domain_state_synthesis(task, domain_state)
            except Exception as e:
                runtime.display.log_event(f"Domain state synthesis failed: {e}")

        # STRATEGY 4: Absolute fallback
        return FinalizationResult(
            final_answer="Unable to complete task. No meaningful progress was made.",
            confidence=0.0,
            synthesis_method="fallback",
            is_partial=True,
        )

    def _llm_synthesis(
        self, task: str, state: "GraphState", runtime: "RuntimeContext"
    ) -> FinalizationResult:
        """Use LLM to synthesize final answer from partial progress.

        Args:
            task: Original task
            state: Current state
            runtime: Runtime context

        Returns:
            FinalizationResult with LLM-synthesized answer
        """
        domain_state = state.get("domain_state")
        draft = state.get("draft_answer")

        # Build context from domain state
        if domain_state:
            state_summary = self._summarize_domain_state(domain_state)
        else:
            state_summary = "No structured progress available."

        prompt = [
            SystemMessage(
                content=(
                    "You are synthesizing a final answer from partial progress.\n"
                    "The task was interrupted before completion. Based on the progress made,\n"
                    "provide the MOST COMPLETE answer possible.\n\n"
                    "Guidelines:\n"
                    "- If enough progress was made to answer the task, provide a full answer\n"
                    "- If progress was insufficient, explain what was accomplished and what remains\n"
                    "- Be honest about confidence level and whether answer is partial\n"
                    "- Do NOT make up information that isn't supported by the progress\n\n"
                    "Output ONLY valid JSON with schema:\n"
                    '{"final_answer": "...", "confidence": 0.0-1.0, "is_partial": true|false}'
                )
            ),
            HumanMessage(
                content=(
                    f"Task: {task}\n\n"
                    f"Progress made:\n{state_summary}\n\n"
                    f"Draft answer (if any): {draft or 'None'}\n\n"
                    "Synthesize the best possible final answer:"
                )
            ),
        ]

        schema = {
            "type": "object",
            "properties": {
                "final_answer": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "is_partial": {"type": "boolean"},
            },
            "required": ["final_answer", "confidence", "is_partial"],
        }

        meta = AgentCallMeta(agent="finalizer", stage="FINALIZER", step_id=0)
        result = self.llm.structured_completion(
            prompt,
            meta=meta,
            model=self.config.model_planner,
            temperature=0.3,
            schema_name="FinalizationSynthesis",
            schema=schema,
            parser=lambda x: x,
            max_output_tokens=512,
        )

        return FinalizationResult(
            final_answer=result.content["final_answer"],
            confidence=result.content["confidence"],
            synthesis_method="llm_synthesis",
            is_partial=result.content["is_partial"],
        )

    def _domain_state_synthesis(self, task: str, domain_state) -> FinalizationResult:
        """Synthesize answer from domain state structured fields.

        Args:
            task: Original task
            domain_state: Domain-specific state

        Returns:
            FinalizationResult with domain-synthesized answer
        """
        if isinstance(domain_state, MathState):
            # Synthesize from math state
            answer_parts = []

            if domain_state.solutions:
                answer_parts.append("Solutions found:")
                for var, val in domain_state.solutions.items():
                    answer_parts.append(f"  {var} = {val}")

            if domain_state.equations:
                answer_parts.append("\nEquations derived:")
                for eq in domain_state.equations[-3:]:  # Last 3 equations
                    answer_parts.append(f"  {eq}")

            if answer_parts:
                return FinalizationResult(
                    final_answer="\n".join(answer_parts),
                    confidence=0.6,
                    synthesis_method="domain_state_math",
                    is_partial=len(domain_state.solutions) == 0,
                )

        elif isinstance(domain_state, CodeState):
            # Synthesize from code state
            answer_parts = []

            if domain_state.functions:
                answer_parts.append(f"Functions defined: {', '.join(domain_state.functions)}")

            if domain_state.tests:
                answer_parts.append(f"Tests created: {', '.join(domain_state.tests)}")

            if domain_state.files:
                answer_parts.append(f"Files referenced: {', '.join(domain_state.files.keys())}")

            if answer_parts:
                return FinalizationResult(
                    final_answer="\n".join(answer_parts),
                    confidence=0.5,
                    synthesis_method="domain_state_code",
                    is_partial=True,
                )

        elif isinstance(domain_state, CreativeState):
            # Synthesize from creative state
            answer_parts = []

            if domain_state.themes:
                answer_parts.append(f"Themes explored: {', '.join(domain_state.themes)}")

            if domain_state.ideas:
                answer_parts.append("\nIdeas generated:")
                for idea in domain_state.ideas[:5]:  # First 5 ideas
                    answer_parts.append(f"  - {idea}")

            if answer_parts:
                return FinalizationResult(
                    final_answer="\n".join(answer_parts),
                    confidence=0.4,
                    synthesis_method="domain_state_creative",
                    is_partial=True,
                )

        # Fallback if domain state doesn't have enough content
        return FinalizationResult(
            final_answer=f"Task: {task}\n\nPartial progress made but insufficient to provide complete answer.",
            confidence=0.2,
            synthesis_method="domain_state_minimal",
            is_partial=True,
        )

    def _summarize_domain_state(self, domain_state) -> str:
        """Create readable summary of domain state progress.

        Args:
            domain_state: Domain-specific state object

        Returns:
            Human-readable summary string
        """
        if isinstance(domain_state, MathState):
            parts = []
            if domain_state.equations:
                parts.append(f"Equations: {len(domain_state.equations)} found")
                parts.append("Latest equations:")
                for eq in domain_state.equations[-3:]:
                    parts.append(f"  {eq}")

            if domain_state.solutions:
                parts.append(f"\nSolutions: {domain_state.solutions}")

            return "\n".join(parts) if parts else "No math progress"

        elif isinstance(domain_state, CodeState):
            parts = []
            if domain_state.functions:
                parts.append(f"Functions: {', '.join(domain_state.functions)}")
            if domain_state.tests:
                parts.append(f"Tests: {', '.join(domain_state.tests)}")
            if domain_state.files:
                parts.append(f"Files: {', '.join(domain_state.files.keys())}")

            return "\n".join(parts) if parts else "No code progress"

        elif isinstance(domain_state, CreativeState):
            parts = []
            if domain_state.themes:
                parts.append(f"Themes: {', '.join(domain_state.themes)}")
            if domain_state.ideas:
                parts.append(f"Ideas ({len(domain_state.ideas)}): {', '.join(domain_state.ideas[:3])}")

            return "\n".join(parts) if parts else "No creative progress"

        else:
            return str(domain_state)
