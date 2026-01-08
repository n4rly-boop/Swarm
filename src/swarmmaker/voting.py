"""Ahead-by-K voting utilities with discriminator functionality.

This module handles consensus-based selection of proposals and solutions
using the First-to-Ahead-by-K algorithm from the MAKER paper.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar

from .schemas import AtomicSolution, DecompositionProposal

T = TypeVar("T")


@dataclass
class VoteOutcome(Generic[T]):
    """Result of a voting round."""

    winner: Optional[T]
    votes: Dict[str, int]
    confident: bool
    total_samples: int = 0


class AheadByKVoter:
    """Counts candidate signatures until one is ahead by K votes.

    Implements the First-to-Ahead-by-K algorithm from the MAKER paper:
    - Continue sampling until winner_count >= max(other_counts) + K
    - Requires minimum sample threshold before declaring confidence
    """

    def __init__(self, ahead_by: int, min_samples: int = 3) -> None:
        """Initialize voter.

        Args:
            ahead_by: Number of votes the winner must lead by.
            min_samples: Minimum samples required before declaring confidence.
        """
        self.ahead_by = max(1, ahead_by)
        self.min_samples = max(1, min_samples)

    def select(
        self,
        candidates: Sequence[T],
        signature_fn: Callable[[T], str],
    ) -> VoteOutcome[T]:
        """Select a winner from candidates using ahead-by-K voting.

        Args:
            candidates: Sequence of candidate objects to vote on.
            signature_fn: Function to extract signature string from candidate.

        Returns:
            VoteOutcome with winner, vote counts, and confidence status.
        """
        if not candidates:
            return VoteOutcome(winner=None, votes={}, confident=False, total_samples=0)

        counts: Counter[str] = Counter()
        by_signature: Dict[str, T] = {}

        for candidate in candidates:
            signature = signature_fn(candidate)
            by_signature[signature] = candidate
            counts[signature] += 1

        total_samples = sum(counts.values())
        leader_sig = self._leader_if_ahead(counts)

        # FIX: Confidence requires BOTH ahead-by-K AND minimum samples
        confident = leader_sig is not None and total_samples >= self.min_samples

        # If we have a leader ahead by K, use them; otherwise use plurality winner
        winner_signature = leader_sig or counts.most_common(1)[0][0]

        return VoteOutcome(
            winner=by_signature.get(winner_signature),
            votes=dict(counts),
            confident=confident,
            total_samples=total_samples,
        )

    def _leader_if_ahead(self, counts: Counter[str]) -> Optional[str]:
        """Check if any candidate is ahead by K votes.

        Args:
            counts: Counter of signature -> vote count.

        Returns:
            Signature of leader if ahead by K, None otherwise.
        """
        if not counts:
            return None

        if len(counts) == 1:
            # Only one candidate - check if it meets minimum samples
            sig = next(iter(counts.keys()))
            if counts[sig] >= self.min_samples:
                return sig
            return None

        top_two = counts.most_common(2)
        (sig_a, count_a), (sig_b, count_b) = top_two

        if count_a - count_b >= self.ahead_by:
            return sig_a

        return None

    def needs_more_samples(self, outcome: VoteOutcome[T]) -> bool:
        """Check if more samples are needed for confidence.

        Args:
            outcome: Previous voting outcome.

        Returns:
            True if more samples would help reach confidence.
        """
        if outcome.confident:
            return False
        if outcome.total_samples >= self.min_samples * 3:
            # Already have plenty of samples, unlikely to converge
            return False
        return True


# ---------------------------------------------------------------------------
# Discriminators (previously in discriminator.py)
# ---------------------------------------------------------------------------


class DecompositionDiscriminator:
    """Votes among decomposition proposals using ahead-by-K."""

    def __init__(self, ahead_by: int, min_samples: int = 3) -> None:
        self.voter = AheadByKVoter(ahead_by, min_samples)

    def select(self, proposals: Sequence[DecompositionProposal]) -> VoteOutcome[DecompositionProposal]:
        return self.voter.select(proposals, lambda p: p.signature)


class SolutionDiscriminator:
    """Votes among atomic solutions using ahead-by-K."""

    def __init__(self, ahead_by: int, min_samples: int = 3) -> None:
        self.voter = AheadByKVoter(ahead_by, min_samples)

    def select(self, solutions: Sequence[AtomicSolution]) -> VoteOutcome[AtomicSolution]:
        return self.voter.select(solutions, lambda s: s.signature)
