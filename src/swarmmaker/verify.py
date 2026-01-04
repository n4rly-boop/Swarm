"""Verifier for SwarmMaker actions."""
import json
from typing import Dict, Iterable, MutableMapping, Optional, Tuple

from .domain_state import MathState
from .math_constraints import MathValidator
from .schemas import Action, PlannerStep


class ActionVerifier:
    """Applies structural checks and mutates shared state once actions are accepted."""

    def __init__(self) -> None:
        self._applied_signatures: set[str] = set()
        self._last_signature: Optional[str] = None
        self._applied_contents: set[str] = set()  # Normalized content for deduplication

    def verify(
        self,
        planner_step: PlannerStep,
        action: Action,
        *,
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        if action.step_id != planner_step.step_id:
            return False, "step mismatch (worker must copy planner step_id exactly)"
        if not action.action_type or not action.action_type.strip():
            return False, "empty action type"
        if not isinstance(action.args, dict):
            return False, "args must be an object"

        action_type = action.action_type.strip()
        if action_type == "final_answer":
            action_type = "FINAL"
            action.action_type = "FINAL"

        if planner_step.stop_condition == "done" and action_type != "FINAL":
            return False, "planner requested done -> must emit FINAL with args.content"

        # Domain-specific validation (replaces string heuristics)
        # Note: domain_state is passed via dry_run context during consensus pre-validation
        # For normal verification, we skip domain validation here and rely on pre-validation
        # This is because we don't have access to domain_state in this method signature

        if action_type == "FINAL":
            content = action.args.get("content")
            if not isinstance(content, str) or not content.strip():
                return False, "FINAL requires args.content as non-empty string"
        elif action_type == "NOTE":
            # Check for content duplication in domain state (anti-stagnation for small models)
            if not dry_run:
                content = action.args.get("content", "")
                if content and self._is_duplicate_content(content, planner_step):
                    return False, "duplicate content (this work was already completed in a previous step)"
        elif action_type == "ASK_CLARIFY":
            # ASK_CLARIFY is discouraged - only allow if args.content explains why it's truly needed
            content = action.args.get("content") or action.args.get("prompt")
            if not isinstance(content, str) or not content.strip():
                return False, "ASK_CLARIFY requires args.content or args.prompt with explanation"
            # Log a warning that this action type is discouraged
            return True, "ASK_CLARIFY accepted but discouraged (consider using NOTE instead)"
        elif action_type == "DO":
            if not action.args:
                return False, "DO requires non-empty args"
            # Check for content duplication (anti-stagnation for small models)
            if not dry_run:
                content = action.args.get("content", "")
                if content and self._is_duplicate_content(content, planner_step):
                    return False, "duplicate content (this work was already completed in a previous step)"
        else:
            return False, f"unknown action_type: {action_type}"

        signature = action.signature

        # Check for duplicates
        if signature == self._last_signature:
            return False, "duplicate signature (same as previous action)"
        if signature in self._applied_signatures:
            return False, "duplicate signature (loop prevention)"

        # Only track signatures during actual verification, not dry-run pre-validation
        if not dry_run:
            self._last_signature = signature
            self._applied_signatures.add(signature)

            # Track normalized content for duplicate detection
            content = action.args.get("content", "")
            if content:
                normalized = self._normalize_content(content)
                if len(normalized) >= 10:  # Only track substantial content
                    self._applied_contents.add(normalized)

        return True, "dry-run accepted" if dry_run else "validated"

    def apply(self, action: Action, state: MutableMapping[str, object]) -> None:
        history = state.setdefault("history_signatures", [])
        if isinstance(history, list):
            history.append(action.signature)

        # Update domain state by merging action
        domain_state = state.get("domain_state")
        if domain_state and hasattr(domain_state, "merge"):
            state["domain_state"] = domain_state.merge(action)

        # Handle FINAL action
        if action.action_type == "FINAL":
            content = self._extract_content(action.args)
            state["draft_answer"] = content
            state["final_answer"] = content
            state["done"] = True

    def applied_signatures(self) -> Iterable[str]:
        return tuple(self._applied_signatures)

    def _normalize_content(self, content: str) -> str:
        """Normalize content for duplicate detection (case-insensitive, punctuation-insensitive)."""
        # Remove common variations that don't change meaning
        normalized = content.lower().strip()
        # Remove trailing punctuation
        normalized = normalized.rstrip('.!,;:')
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized

    def _is_duplicate_content(self, content: str, planner_step: PlannerStep) -> bool:
        """Check if this content was already submitted in a previous action."""
        normalized = self._normalize_content(content)

        # Skip very short content (likely just step descriptions)
        if len(normalized) < 10:
            return False

        return normalized in self._applied_contents

    def _extract_content(self, args: Dict[str, object]) -> str:
        content = args.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return json.dumps(args, ensure_ascii=False)
