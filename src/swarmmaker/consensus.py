"""Consensus helpers for SwarmMaker voters."""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .schemas import Action


@dataclass
class ConsensusResult:
    """Outcome of a consensus round."""

    selected: Optional[Action]
    needs_judge: bool
    top_candidates: List[Action]
    votes: Dict[str, int]
    early_stop_signature: Optional[str] = None


class ConsensusEngine:
    """Aggregates voter actions with ahead-by-K early stop."""

    def __init__(self, ahead_by: int) -> None:
        self.ahead_by = max(1, ahead_by)

    def leader_if_ahead(self, votes: Dict[str, int]) -> Optional[str]:
        """Return winning signature if ahead-by constraint holds."""

        if not votes:
            return None
        sorted_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        if len(sorted_votes) == 1:
            return sorted_votes[0][0]
        (sig_a, count_a), (_, count_b) = sorted_votes[:2]
        if count_a - count_b >= self.ahead_by:
            return sig_a
        return None

    def decide(self, actions: Sequence[Action]) -> ConsensusResult:
        if not actions:
            return ConsensusResult(
                selected=None,
                needs_judge=False,
                top_candidates=[],
                votes={},
                early_stop_signature=None,
            )
        signature_to_action: Dict[str, Action] = {}
        counts: Dict[str, int] = Counter()
        for action in actions:
            signature_to_action[action.signature] = action
            counts[action.signature] += 1
        leader_signature = self.leader_if_ahead(counts)
        sorted_signatures = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                -(signature_to_action[item[0]].confidence or 0.0),
                item[0],
            ),
        )
        top_actions: List[Action] = []
        for signature, _ in sorted_signatures:
            top_actions.append(signature_to_action[signature])
            if len(top_actions) == 2:
                break
        needs_judge = leader_signature is None and len(sorted_signatures) > 1
        selected = signature_to_action.get(sorted_signatures[0][0]) if not needs_judge else None
        return ConsensusResult(
            selected=selected,
            needs_judge=needs_judge,
            top_candidates=top_actions if needs_judge else [],
            votes=dict(counts),
            early_stop_signature=leader_signature,
        )
