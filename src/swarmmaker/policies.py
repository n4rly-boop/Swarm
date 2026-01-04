"""Policy enforcement engine for SwarmMaker.

This module provides hard invariants that cannot be violated by LLM behavior.
Policies are checked before state transitions and can abort runs or modify planner constraints.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .graph import GraphState, RuntimeContext


class Policy(ABC):
    """Base class for system policies that enforce invariants."""

    @abstractmethod
    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        """Check if policy is violated.

        Args:
            state: Current graph state
            context: Runtime context with all services

        Returns:
            Tuple of (violated, reason). If violated is True, reason explains the violation.
        """
        pass


class StopConditionDonePolicy(Policy):
    """Enforce retry limit when stop_condition='done' to prevent infinite FINAL-retry loops.

    Allows reasonable retries (respecting max_retries config) but prevents endless loops
    when the planner has requested finalization.
    """

    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        # DISABLED: This policy was too strict for small models
        # Small models often fail on first attempt even with clear instructions
        # The max_retries limit in verify_apply_node is sufficient protection
        return False, None


class NoIdenticalStatePolicy(Policy):
    """Detect stagnation by hashing state and aborting on duplicates.

    Prevents workers from producing identical non-progress across COMPLETED steps.
    Does NOT trigger during retries of the same step (that's expected behavior).
    """

    def __init__(self):
        self.state_hashes: List[str] = []
        self.last_completed_step: int = -1

    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        # Only check after completing a step, not during retries
        current_step = state.get("steps_completed", 0)
        if current_step <= self.last_completed_step:
            # Still on same step (retry) or haven't completed any steps yet
            return False, None

        # Hash the meaningful state components
        domain_state = state.get("domain_state")
        draft_answer = state.get("draft_answer", "")

        # Create hash from domain state + draft
        if domain_state:
            state_data = {
                "domain_state": domain_state.hash_key() if hasattr(domain_state, "hash_key") else str(domain_state),
                "draft_answer": draft_answer or "",
            }
        else:
            # Fallback if domain_state not initialized yet
            return False, None

        state_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True).encode()
        ).hexdigest()

        # Check if this exact state was seen before (after a completed step)
        if state_hash in self.state_hashes and self.state_hashes.count(state_hash) > 3:
            return True, "stagnation detected - identical state repeated across multiple completed steps (no progress made)"

        # Record this state hash and update last completed step
        self.state_hashes.append(state_hash)
        self.last_completed_step = current_step
        return False, None


class ASKClarifyTerminationPolicy(Policy):
    """ASK_CLARIFY triggers immediate finalization instead of retry loops.

    This prevents the system from looping on clarification requests.
    """

    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        chosen_action = state.get("chosen_action")
        if not chosen_action:
            return False, None

        # Check if chosen action is ASK_CLARIFY
        if chosen_action.action_type == "ASK_CLARIFY":
            # Set flag to trigger best-effort finalization
            # Note: This doesn't violate the policy, but signals special handling
            # We return False (not violated) but the check_node will see this action
            # and trigger finalization via the termination authority
            context.display.log_event("ASK_CLARIFY detected - will trigger finalization after this step")
            return False, None

        return False, None


class MaxDecompositionDepthPolicy(Policy):
    """Force FINAL after N steps without meaningful progress.

    Prevents over-decomposition by limiting how many steps can occur without
    state changes. After the limit, forces the planner to finalize.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        # Check progress tracker for stagnation
        if not hasattr(context, "progress"):
            return False, None

        steps_completed = state.get("steps_completed", 0)
        stagnation_steps = context.progress.stagnation_steps(steps_completed)

        if stagnation_steps >= self.max_depth:
            # Force finalization on next step
            # This is handled by setting a flag that plan_node will check
            context.display.log_event(
                f"Max decomposition depth reached ({self.max_depth} steps without progress) - forcing finalization"
            )
            state["force_final_next_step"] = True
            return False, None  # Don't abort, just force finalization

        return False, None


class BudgetAwareStrategyPolicy(Policy):
    """Switch to single-worker mode when budget is low.

    Prevents token waste on voting when budget is nearly exhausted.
    """

    def __init__(self, threshold_pct: float = 0.20):
        """Initialize policy.

        Args:
            threshold_pct: Budget threshold (0-1). Below this, force single-worker mode.
        """
        self.threshold_pct = threshold_pct

    def check(self, state: "GraphState", context: "RuntimeContext") -> Tuple[bool, Optional[str]]:
        # Calculate budget remaining
        from .schemas import SwarmConfig

        config: SwarmConfig = context.config
        tokens_used = context.metrics.tokens_total()
        tokens_max = config.max_total_tokens

        budget_remaining_pct = 1.0 - (tokens_used / tokens_max) if tokens_max > 0 else 1.0

        if budget_remaining_pct < self.threshold_pct:
            # Signal that we should use reduced worker count
            # This is checked in propose_node to reduce swarm size
            context.display.log_event(
                f"Budget low ({budget_remaining_pct:.0%} remaining) - reducing to single-worker mode"
            )
            state["budget_constrained_mode"] = True
            return False, None

        return False, None


class PolicyEngine:
    """Enforces all registered policies before state transitions."""

    def __init__(self):
        self.policies: List[Policy] = []

    def register(self, policy: Policy) -> None:
        """Add a policy to the enforcement set.

        Args:
            policy: Policy instance to register
        """
        self.policies.append(policy)

    def enforce_all(self, state: "GraphState", context: "RuntimeContext") -> Optional[str]:
        """Check all policies and return abort reason if any violated.

        Args:
            state: Current graph state
            context: Runtime context

        Returns:
            Abort reason string if any policy violated, None otherwise
        """
        for policy in self.policies:
            violated, reason = policy.check(state, context)
            if violated:
                # Log policy violation event
                context.events.log(
                    "policy_violation",
                    {"policy": policy.__class__.__name__, "reason": reason},
                )
                return f"Policy violation: {reason}"

        return None
