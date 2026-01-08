"""Red-flag guard for atomic solutions.

Based on the MAKER paper's finding that outputs >700 tokens have ~90% error rate.
This module implements strict rejection filters applied before voting.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .schemas import AtomicSolution, ThresholdConfig


@dataclass
class RedFlagRejection:
    """Structured rejection payload for observability."""

    pattern: str
    reason: str


class RedFlagGuard:
    """Filters atomic solutions matching high-risk patterns.

    From MAKER paper:
    - Outputs exceeding ~700 tokens correlate with ~90% error rate
    - Formatting errors indicate reasoning confusion
    - Rejected outputs are discarded silently (no repair)
    """

    def __init__(self, thresholds: Optional[ThresholdConfig] = None) -> None:
        """Initialize guard with threshold configuration.

        Args:
            thresholds: Centralized threshold config. Uses defaults if None.
        """
        self.thresholds = thresholds or ThresholdConfig()

        # Compile patterns once
        self._meta_chatter = re.compile(
            r"\b(?:as an ai|i (?:think|believe)|let me|i will now)\b",
            re.IGNORECASE,
        )
        self._placeholder = re.compile(
            r"(?:TODO|TBD|fill in|<your answer>|insert answer here|\[answer\]|\{answer\})",
            re.IGNORECASE,
        )
        self._self_contradiction = re.compile(
            r"\b(?:cannot|can't|unable)\b[^.]{0,80}\b(?:but|however)\b[^.]{0,80}\b(?:can|able|will)\b",
            re.IGNORECASE,
        )

    def inspect(self, solution: AtomicSolution) -> Tuple[bool, Optional[RedFlagRejection]]:
        """Inspect a solution for red-flag patterns.

        Args:
            solution: Atomic solution to inspect.

        Returns:
            Tuple of (passed, rejection). If passed is True, rejection is None.
        """
        text = f"{solution.solution.strip()}\n{solution.work_shown.strip()}".strip()

        # Check for empty content
        if not text:
            return False, RedFlagRejection("missing_field", "solution and work_shown must be non-empty")

        # Strict token limit (paper: >700 tokens = ~90% error)
        estimated_tokens = len(text) / 4  # Rough estimate: 4 chars per token
        if estimated_tokens > self.thresholds.max_solution_tokens:
            return False, RedFlagRejection(
                "excessive_tokens",
                f"estimated {int(estimated_tokens)} tokens exceeds {self.thresholds.max_solution_tokens} limit",
            )

        # Character limit backup
        if len(text) > self.thresholds.max_solution_chars:
            return False, RedFlagRejection(
                "excessive_length",
                f"content exceeds {self.thresholds.max_solution_chars} characters",
            )

        # Repetition detection
        if self._has_repetition(text):
            return False, RedFlagRejection("repetition", "content repeats the same phrase multiple times")

        # Meta-commentary detection
        if self._meta_chatter.search(text):
            return False, RedFlagRejection(
                "meta_chatter",
                "meta commentary detected (\"I think\", \"As an AI\", etc.)",
            )

        # Placeholder detection
        if self._placeholder.search(text) or text.count("...") >= self.thresholds.repetition_threshold:
            return False, RedFlagRejection(
                "placeholder",
                "placeholder text detected (TODO / ... / <your answer>)",
            )

        # Self-contradiction detection
        if self._self_contradiction.search(text):
            return False, RedFlagRejection(
                "self_contradiction",
                "statement contradicts itself (e.g., 'I cannot... but I can')",
            )

        # Low confidence
        if solution.confidence < self.thresholds.min_confidence:
            return False, RedFlagRejection(
                "low_confidence",
                f"confidence {solution.confidence:.2f} below threshold {self.thresholds.min_confidence}",
            )

        return True, None

    def inspect_batch(self, solutions: List[AtomicSolution]) -> Tuple[List[AtomicSolution], List[RedFlagRejection]]:
        """Inspect a batch of solutions, returning passed and rejections.

        Args:
            solutions: List of solutions to inspect.

        Returns:
            Tuple of (passed_solutions, rejections).
        """
        passed = []
        rejections = []

        for solution in solutions:
            ok, rejection = self.inspect(solution)
            if ok:
                passed.append(solution)
            elif rejection:
                rejections.append(rejection)

        return passed, rejections

    def _has_repetition(self, text: str) -> bool:
        """Check for repetitive patterns in text.

        Args:
            text: Text to check.

        Returns:
            True if repetitive patterns detected.
        """
        threshold = self.thresholds.repetition_threshold

        # Check for repeated lines
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= threshold:
                return True

        # Check for repeated token windows
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < 4:
            return False

        window: List[str] = []
        seen: dict[tuple[str, ...], int] = {}

        for token in tokens:
            window.append(token)
            if len(window) > 4:
                window.pop(0)
            if len(window) < 2:
                continue
            key = tuple(window)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= threshold:
                return True

        return False
