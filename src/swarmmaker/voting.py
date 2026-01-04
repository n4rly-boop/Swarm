"""Ahead-by-K voting utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class VoteOutcome(Generic[T]):
    winner: Optional[T]
    votes: Dict[str, int]
    confident: bool


class AheadByKVoter:
    """Counts candidate signatures until one is ahead by K votes."""

    def __init__(self, ahead_by: int) -> None:
        self.ahead_by = max(1, ahead_by)

    def select(self, candidates: Sequence[T], signature_fn: Callable[[T], str]) -> VoteOutcome[T]:
        if not candidates:
            return VoteOutcome(winner=None, votes={}, confident=False)
        counts: Counter[str] = Counter()
        by_signature: Dict[str, T] = {}
        for candidate in candidates:
            signature = signature_fn(candidate)
            by_signature[signature] = candidate
            counts[signature] += 1

        leader = counts.most_common(1)[0][0]
        confident = self._leader_if_ahead(counts) is not None or len(counts) == 1
        winner_signature = self._leader_if_ahead(counts) or leader
        return VoteOutcome(
            winner=by_signature.get(winner_signature),
            votes=dict(counts),
            confident=confident,
        )

    def _leader_if_ahead(self, counts: Counter[str]) -> Optional[str]:
        if not counts:
            return None
        if len(counts) == 1:
            return next(iter(counts.keys()))
        top_two = counts.most_common(2)
        (sig_a, count_a), (sig_b, count_b) = top_two
        if count_a - count_b >= self.ahead_by:
            return sig_a
        return None
