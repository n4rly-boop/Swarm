"""Consensus helpers for SwarmMaker voters."""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .schemas import Action


@dataclass
class ConsensusResult:
    """Outcome of a consensus pass."""

    selected: Optional[Action]
    needs_judge: bool
    top_candidates: List[Action]
    votes: Dict[str, int]


class ConsensusEngine:
    """Aggregates voter actions and enforces ahead-by early stop."""

    def __init__(self, ahead_by: int) -> None:
        self.ahead_by = max(1, ahead_by)

    def decide(self, actions: List[Action]) -> ConsensusResult:
        if not actions:
            return ConsensusResult(
                selected=None,
                needs_judge=False,
                top_candidates=[],
                votes={},
            )

        counts = Counter(action.signature for action in actions)
        sorted_actions = sorted(
            actions,
            key=lambda act: (counts[act.signature], act.confidence or 0.0),
            reverse=True,
        )
        winner = sorted_actions[0]
        winner_votes = counts[winner.signature]
        needs_judge = winner_votes < self.ahead_by and len(sorted_actions) > 1
        top_candidates: List[Action] = sorted_actions[:2] if needs_judge else []
        votes = {action.signature: counts[action.signature] for action in actions}

        return ConsensusResult(
            selected=winner if not needs_judge else None,
            needs_judge=needs_judge,
            top_candidates=top_candidates,
            votes=votes,
        )
