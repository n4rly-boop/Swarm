"""Verifier for SwarmMaker actions."""
from typing import Iterable, Tuple

from .schemas import Action, PlannerStep


class ActionVerifier:
    """Applies static checks to candidate actions."""

    def __init__(self) -> None:
        self._applied_signatures: dict[int, set[str]] = {}

    def verify(
        self,
        planner_step: PlannerStep,
        action: Action,
        *,
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        """Return (ok, reason)."""

        if action.step_id != planner_step.step_id:
            return False, "step mismatch"

        if not action.action_type or not action.action_type.strip():
            return False, "empty action type"

        if not isinstance(action.args, dict):
            return False, "args must be an object"

        if not action.args:
            return False, "args cannot be empty"

        if self._is_duplicate(planner_step.step_id, action.signature):
            return False, "duplicate signature for this step"

        if dry_run:
            # In dry-run we only simulate application, still record signature.
            self._mark_signature(planner_step.step_id, action.signature)
            return True, "dry-run accepted"

        # Placeholder for real-world validation hooks.
        self._mark_signature(planner_step.step_id, action.signature)
        return True, "validated"

    def _is_duplicate(self, step_id: int, signature: str) -> bool:
        return signature in self._applied_signatures.get(step_id, set())

    def _mark_signature(self, step_id: int, signature: str) -> None:
        self._applied_signatures.setdefault(step_id, set()).add(signature)

    def applied_signatures(self, step_id: int) -> Iterable[str]:
        return tuple(self._applied_signatures.get(step_id, set()))
