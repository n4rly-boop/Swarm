"""System-driven termination authority for SwarmMaker.

This module provides system-level decision-making about when to terminate runs,
overriding LLM decisions when necessary to prevent infinite loops and guarantee progress.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .router import OrchestrationStrategy

if TYPE_CHECKING:
    from .graph import GraphState, RuntimeContext
    from .schemas import SwarmConfig


class TerminationReason(str, Enum):
    """Reasons for system-initiated termination."""

    SUCCESS = "success"
    STAGNATION = "stagnation"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS = "max_steps"
    POLICY_VIOLATION = "policy_violation"
    FORCED_FINALIZATION = "forced_finalization"


@dataclass
class TerminationDecision:
    """System decision about whether and how to terminate.

    Attributes:
        should_terminate: If True, run should end after this decision
        reason: Why termination was triggered (if applicable)
        force_final: If True, force FINAL action generation without aborting
        message: Human-readable explanation of decision
    """

    should_terminate: bool
    reason: Optional[TerminationReason]
    force_final: bool
    message: str


class TerminationAuthority:
    """Decides when runs should terminate based on system state.

    This is the final authority on termination - it can override LLM decisions
    to prevent infinite loops, enforce budget limits, and guarantee progress.
    """

    def __init__(self, config: "SwarmConfig"):
        """Initialize termination authority.

        Args:
            config: Swarm configuration with budget limits
        """
        self.config = config

    def decide(self, state: "GraphState", runtime: "RuntimeContext") -> TerminationDecision:
        """Make termination decision based on current state.

        Args:
            state: Current graph state
            runtime: Runtime context with all services

        Returns:
            TerminationDecision indicating whether and how to terminate
        """
        # RULE 1: Success - FINAL action already applied
        if state.get("done"):
            return TerminationDecision(
                should_terminate=True,
                reason=TerminationReason.SUCCESS,
                force_final=False,
                message="Task completed successfully",
            )

        # RULE 2: Budget exceeded
        tokens_used = runtime.metrics.tokens_total()
        tokens_max = self.config.max_total_tokens

        if tokens_used >= tokens_max:
            return TerminationDecision(
                should_terminate=True,
                reason=TerminationReason.BUDGET_EXCEEDED,
                force_final=True,
                message=f"Token budget exceeded ({tokens_used}/{tokens_max}) - forcing finalization",
            )

        # RULE 3: Max steps reached
        steps_completed = state.get("steps_completed", 0)
        if steps_completed >= self.config.max_steps:
            return TerminationDecision(
                should_terminate=True,
                reason=TerminationReason.MAX_STEPS,
                force_final=True,
                message=f"Max steps reached ({steps_completed}/{self.config.max_steps}) - forcing finalization",
            )

        # RULE 4: Stagnation detected (3 steps without progress)
        if hasattr(runtime, "progress"):
            stagnation_steps = runtime.progress.stagnation_steps(steps_completed)
            if stagnation_steps >= 3:
                return TerminationDecision(
                    should_terminate=False,  # Don't abort yet
                    reason=None,
                    force_final=True,  # But force FINAL action
                    message=f"Stagnation detected ({stagnation_steps} steps without progress) - forcing final answer synthesis",
                )

        # RULE 5: Single-shot strategy completed
        task_analysis = state.get("task_analysis")
        if task_analysis and task_analysis.strategy == OrchestrationStrategy.SINGLE_SHOT:
            if steps_completed >= 1:
                return TerminationDecision(
                    should_terminate=True,
                    reason=TerminationReason.SUCCESS,
                    force_final=False,
                    message="Single-shot strategy completed (1 step)",
                )

        # RULE 6: ASK_CLARIFY action chosen
        chosen_action = state.get("chosen_action")
        if chosen_action and chosen_action.action_type == "ASK_CLARIFY":
            return TerminationDecision(
                should_terminate=False,
                reason=None,
                force_final=True,
                message="ASK_CLARIFY action detected - forcing finalization instead of clarification loop",
            )

        # No termination needed
        return TerminationDecision(
            should_terminate=False, reason=None, force_final=False, message="Continue execution"
        )
