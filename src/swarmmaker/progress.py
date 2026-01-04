"""Progress tracking and stagnation detection for SwarmMaker.

This module detects when the system is making meaningful progress vs. spinning wheels.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from .graph import GraphState


@dataclass
class StateSnapshot:
    """Immutable snapshot of meaningful state at a point in time."""

    domain_state_hash: str
    draft_answer_hash: str
    equations: Set[str]  # Extracted equations from domain state
    facts: Set[str]  # Extracted facts/content
    step_id: int


class ProgressTracker:
    """Detects progress and stagnation by tracking state changes."""

    def __init__(self):
        self.history: List[StateSnapshot] = []
        self.last_progress_step: int = 0

    def snapshot(self, state: "GraphState") -> StateSnapshot:
        """Create immutable snapshot of current state.

        Args:
            state: Current graph state

        Returns:
            StateSnapshot with hashes and extracted content
        """
        domain_state = state.get("domain_state")
        draft = state.get("draft_answer", "")
        step_id = state.get("steps_completed", 0)

        # Extract structured content
        equations = self._extract_equations(domain_state)
        facts = self._extract_facts(domain_state)

        # Create hashes
        if domain_state and hasattr(domain_state, "hash_key"):
            domain_hash = domain_state.hash_key()
        elif domain_state:
            domain_hash = hashlib.sha256(
                json.dumps(str(domain_state), sort_keys=True).encode()
            ).hexdigest()
        else:
            domain_hash = hashlib.sha256(b"").hexdigest()

        draft_hash = hashlib.sha256((draft or "").encode()).hexdigest()

        return StateSnapshot(
            domain_state_hash=domain_hash,
            draft_answer_hash=draft_hash,
            equations=equations,
            facts=facts,
            step_id=step_id,
        )

    def measure_progress(self, current: StateSnapshot) -> float:
        """Measure progress delta from last snapshot.

        Args:
            current: Current state snapshot

        Returns:
            Progress delta: 0.0 = no change, 1.0 = major change
        """
        if not self.history:
            # First snapshot always counts as progress
            self.last_progress_step = current.step_id
            return 1.0

        prev = self.history[-1]

        # Check for state changes
        domain_changed = current.domain_state_hash != prev.domain_state_hash
        draft_changed = current.draft_answer_hash != prev.draft_answer_hash
        state_changed = domain_changed or draft_changed

        # Check for content enrichment
        new_equations = len(current.equations - prev.equations)
        new_facts = len(current.facts - prev.facts)

        # Determine progress level
        if state_changed and (new_equations > 0 or new_facts > 0):
            # State changed AND new content added = major progress
            self.last_progress_step = current.step_id
            return 1.0
        elif state_changed:
            # State changed but no new structured content = minor progress
            self.last_progress_step = current.step_id
            return 0.5
        else:
            # No state change = no progress (stagnation)
            return 0.0

    def stagnation_steps(self, current_step: int) -> int:
        """Count how many steps have passed since last progress.

        Args:
            current_step: Current step number

        Returns:
            Number of steps since last meaningful progress
        """
        return current_step - self.last_progress_step

    def _extract_equations(self, domain_state) -> Set[str]:
        """Extract equations from domain state.

        Args:
            domain_state: Domain-specific state object

        Returns:
            Set of normalized equation strings
        """
        equations = set()

        if not domain_state:
            return equations

        # Try to extract from MathState if available
        if hasattr(domain_state, "equations"):
            equations.update(domain_state.equations)

        # Also check for equations in other fields
        if hasattr(domain_state, "intermediate_results"):
            for result in domain_state.intermediate_results:
                # Find patterns like "x^2 - 7x + 6 = 0"
                matches = re.findall(r'[^=]+=+[^=]+', str(result))
                equations.update(m.strip() for m in matches)

        return equations

    def _extract_facts(self, domain_state) -> Set[str]:
        """Extract facts from domain state.

        Args:
            domain_state: Domain-specific state object

        Returns:
            Set of normalized fact strings
        """
        facts = set()

        if not domain_state:
            return facts

        # Extract based on domain type
        if hasattr(domain_state, "intermediate_results"):
            # MathState
            facts.update(str(r).strip().lower() for r in domain_state.intermediate_results if r)

        if hasattr(domain_state, "solutions"):
            # MathState solutions
            for var, val in getattr(domain_state, "solutions", {}).items():
                facts.add(f"{var}={val}".lower())

        if hasattr(domain_state, "ideas"):
            # CreativeState
            facts.update(str(idea).strip().lower() for idea in domain_state.ideas if idea)

        if hasattr(domain_state, "functions"):
            # CodeState
            facts.update(str(func).strip().lower() for func in domain_state.functions if func)

        return facts
