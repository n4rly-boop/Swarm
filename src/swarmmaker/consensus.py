"""Consensus helpers for SwarmMaker voters."""
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .schemas import Action

if TYPE_CHECKING:
    from .schemas import PlannerStep
    from .verify import ActionVerifier


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

    def __init__(self, ahead_by: int, verifier: Optional["ActionVerifier"] = None) -> None:
        self.ahead_by = max(1, ahead_by)
        self.verifier = verifier

    def leader_if_ahead(
        self,
        votes: Dict[str, int],
        signature_to_action: Optional[Dict[str, Action]] = None,
        planner_step: Optional["PlannerStep"] = None,
    ) -> Optional[str]:
        """Return winning signature if ahead-by constraint holds AND passes verification.

        Args:
            votes: Vote counts by signature
            signature_to_action: Mapping from signature to action (for pre-validation)
            planner_step: Current planner step (for pre-validation)

        Returns:
            Winning signature if ahead and valid, None otherwise
        """
        if not votes:
            return None
        sorted_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))

        # Single candidate
        if len(sorted_votes) == 1:
            sig = sorted_votes[0][0]
            # Pre-validate if verifier available
            if self.verifier and signature_to_action and planner_step and sig in signature_to_action:
                action = signature_to_action[sig]
                ok, _ = self.verifier.verify(planner_step, action, dry_run=True)
                return sig if ok else None
            return sig

        # Check if leader is ahead by K votes
        (sig_a, count_a), (_, count_b) = sorted_votes[:2]
        if count_a - count_b >= self.ahead_by:
            # Pre-validate leader before declaring consensus
            if self.verifier and signature_to_action and planner_step and sig_a in signature_to_action:
                action = signature_to_action[sig_a]
                ok, _ = self.verifier.verify(planner_step, action, dry_run=True)
                return sig_a if ok else None
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
