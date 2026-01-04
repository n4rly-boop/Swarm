"""Verifier for SwarmMaker actions."""
import json
from typing import Dict, Iterable, MutableMapping, Optional, Tuple

from .schemas import Action, PlannerStep


class ActionVerifier:
    """Applies structural checks and mutates shared state once actions are accepted."""

    def __init__(self) -> None:
        self._applied_signatures: set[str] = set()
        self._last_signature: Optional[str] = None

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

        if action_type == "FINAL":
            content = action.args.get("content")
            if not isinstance(content, str) or not content.strip():
                return False, "FINAL requires args.content as non-empty string"
        elif action_type in ("NOTE", "ASK_CLARIFY"):
            pass
        elif action_type == "DO":
            if not action.args:
                return False, "DO requires non-empty args"
        else:
            return False, f"unknown action_type: {action_type}"

        signature = action.signature
        if signature == self._last_signature:
            return False, "duplicate signature (same as previous action)"
        if signature in self._applied_signatures:
            return False, "duplicate signature (loop prevention)"

        self._last_signature = signature
        self._applied_signatures.add(signature)
        return True, "dry-run accepted" if dry_run else "validated"

    def apply(self, action: Action, state: MutableMapping[str, object]) -> None:
        history = state.setdefault("history_signatures", [])
        if isinstance(history, list):
            history.append(action.signature)
        notes = state.setdefault("notes", [])
        if not isinstance(notes, list):
            raise TypeError("state.notes must remain a list")
        if action.action_type == "NOTE":
            note = self._extract_content(action.args)
            notes.append(note)
        elif action.action_type == "DO":
            description = action.args.get("description")
            if isinstance(description, str) and description.strip():
                notes.append(f"Did: {description.strip()}")
            else:
                notes.append(f"Did: {json.dumps(action.args, ensure_ascii=False)}")
        elif action.action_type == "ASK_CLARIFY":
            question = self._extract_content(action.args)
            notes.append(f"Clarify: {question}")
        elif action.action_type == "FINAL":
            content = self._extract_content(action.args)
            state["draft_answer"] = content
            state["final_answer"] = content
            state["done"] = True

    def applied_signatures(self) -> Iterable[str]:
        return tuple(self._applied_signatures)

    def _extract_content(self, args: Dict[str, object]) -> str:
        content = args.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return json.dumps(args, ensure_ascii=False)
