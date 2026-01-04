"""Task classification and orchestration strategy selection for SwarmMaker.

This module routes tasks to appropriate orchestration strategies based on
task characteristics, preventing over-decomposition and optimizing resource usage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import AgentCallMeta

if TYPE_CHECKING:
    from .llm import LLMClient
    from .schemas import SwarmConfig


class TaskCategory(str, Enum):
    """Categories of tasks for routing decisions."""

    MATH = "math"
    CODE = "code"
    CREATIVE = "creative"
    FACTUAL = "factual"
    AMBIGUOUS = "ambiguous"


class OrchestrationStrategy(str, Enum):
    """Orchestration strategies for different task types."""

    SINGLE_SHOT = "single_shot"  # No decomposition, single strong model
    DETERMINISTIC = "deterministic"  # Decompose but no voting, single solver
    CONSENSUS = "consensus"  # Full swarm with voting
    MATH_CONSTRAINED = "math_constrained"  # Domain-specific for math


@dataclass
class TaskAnalysis:
    """Result of task classification and strategy selection."""

    category: TaskCategory
    strategy: OrchestrationStrategy
    confidence: float
    estimated_steps: int
    requires_creativity: bool
    is_well_defined: bool


class TaskRouter:
    """Classifies tasks and selects appropriate orchestration strategy."""

    def __init__(self, llm_client: "LLMClient", config: "SwarmConfig"):
        """Initialize task router.

        Args:
            llm_client: LLM client for task classification
            config: Swarm configuration
        """
        self.llm = llm_client
        self.config = config

    def analyze(self, task: str) -> TaskAnalysis:
        """Classify task and select orchestration strategy.

        Args:
            task: Task description to analyze

        Returns:
            TaskAnalysis with category, strategy, and metadata
        """
        # Use LLM to classify task characteristics
        classification = self._classify_task(task)

        # Apply routing rules to select strategy
        strategy = self._select_strategy(classification, task)

        return TaskAnalysis(
            category=TaskCategory(classification["category"]),
            strategy=strategy,
            confidence=classification["confidence"],
            estimated_steps=classification["estimated_steps"],
            requires_creativity=classification["requires_creativity"],
            is_well_defined=classification["is_well_defined"],
        )

    def _classify_task(self, task: str) -> Dict[str, Any]:
        """Use LLM to classify task characteristics.

        Args:
            task: Task description

        Returns:
            Classification dictionary with category, estimated_steps, etc.
        """
        prompt = [
            SystemMessage(
                content=(
                    "Classify this task along these dimensions:\n"
                    "1. category: Choose the BEST match from [math, code, creative, factual, ambiguous]\n"
                    "   - math: Mathematical problems, equations, calculations\n"
                    "   - code: Programming, software development, debugging\n"
                    "   - creative: Writing, brainstorming, open-ended ideation\n"
                    "   - factual: Information lookup, summarization, fact-based QA\n"
                    "   - ambiguous: Unclear requirements or mixed categories\n"
                    "2. estimated_steps: How many atomic steps needed? (1-20)\n"
                    "   - 1-2: Single straightforward operation\n"
                    "   - 3-5: Few distinct sub-tasks\n"
                    "   - 6+: Complex multi-step process\n"
                    "3. requires_creativity: true if multiple valid approaches/subjective judgment\n"
                    "4. is_well_defined: true if clear success criteria exist\n"
                    "5. confidence: Your confidence in this classification (0-1)\n"
                    "\nOutput ONLY valid JSON matching the schema."
                )
            ),
            HumanMessage(content=f"Task: {task}"),
        ]

        schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["math", "code", "creative", "factual", "ambiguous"],
                },
                "estimated_steps": {"type": "integer", "minimum": 1, "maximum": 20},
                "requires_creativity": {"type": "boolean"},
                "is_well_defined": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "category",
                "estimated_steps",
                "requires_creativity",
                "is_well_defined",
                "confidence",
            ],
        }

        meta = AgentCallMeta(agent="router", stage="ROUTER", step_id=0)
        result = self.llm.structured_completion(
            prompt,
            meta=meta,
            model=self.config.model_planner,
            temperature=0.2,
            schema_name="TaskClassification",
            schema=schema,
            parser=lambda x: x,
            max_output_tokens=256,
        )

        return result.content

    def _select_strategy(self, classification: Dict[str, Any], task: str) -> OrchestrationStrategy:
        """Apply routing rules to select orchestration strategy.

        Args:
            classification: LLM classification results
            task: Original task string

        Returns:
            Selected OrchestrationStrategy
        """
        category = classification["category"]
        estimated_steps = classification["estimated_steps"]
        requires_creativity = classification["requires_creativity"]
        is_well_defined = classification["is_well_defined"]

        # RULE 1: Single-shot for simple, well-defined tasks
        if estimated_steps <= 2 and is_well_defined and not requires_creativity:
            return OrchestrationStrategy.SINGLE_SHOT

        # RULE 2: Math-specific strategy for math tasks
        if category == "math":
            return OrchestrationStrategy.MATH_CONSTRAINED

        # RULE 3: Deterministic for well-defined, non-creative tasks
        if is_well_defined and not requires_creativity and category in ["factual", "code"]:
            return OrchestrationStrategy.DETERMINISTIC

        # RULE 4: Consensus for creative or ambiguous tasks
        if requires_creativity or category in ["creative", "ambiguous"]:
            return OrchestrationStrategy.CONSENSUS

        # DEFAULT: Consensus (safest fallback)
        return OrchestrationStrategy.CONSENSUS
