"""Red-flag guard for atomic solutions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .schemas import AtomicSolution


@dataclass
class RedFlagRejection:
    """Structured rejection payload for observability."""

    pattern: str
    reason: str


class RedFlagGuard:
    """Filters atomic solutions matching high-risk patterns from CLAUDE.md."""

    def __init__(
        self,
        *,
        max_chars: int = 4000,
        min_confidence: float = 0.3,
        repetition_threshold: int = 3,
    ) -> None:
        self.max_chars = max_chars
        self.min_confidence = min_confidence
        self.repetition_threshold = repetition_threshold
        self._meta_chatter = re.compile(
            r"\b(?:as an ai|i (?:think|believe)|let me|i will now)\b", re.IGNORECASE
        )
        self._placeholder = re.compile(
            r"(?:TODO|TBD|fill in|<your answer>|insert answer here)", re.IGNORECASE
        )
        self._self_contradiction = re.compile(
            r"\b(?:cannot|can't|unable)\b[^.]{0,80}\b(?:but|however)\b[^.]{0,80}\b(?:can|able|will)\b",
            re.IGNORECASE,
        )

    def inspect(self, solution: AtomicSolution) -> Tuple[bool, Optional[RedFlagRejection]]:
        text = f"{solution.solution.strip()}\n{solution.work_shown.strip()}".strip()
        if not text:
            return False, RedFlagRejection("missing_field", "solution and work_shown must be non-empty")

        if len(text) > self.max_chars:
            return False, RedFlagRejection("excessive_length", f"content exceeds {self.max_chars} characters")

        if self._has_repetition(text):
            return False, RedFlagRejection("repetition", "content repeats the same phrase multiple times")

        if self._meta_chatter.search(text):
            return False, RedFlagRejection("meta_chatter", "meta commentary detected (\"I think\", \"As an AI\", etc.)")

        if self._placeholder.search(text) or text.count("...") >= self.repetition_threshold:
            return False, RedFlagRejection("placeholder", "placeholder text detected (TODO / ... / <your answer>)")

        if self._self_contradiction.search(text):
            return False, RedFlagRejection(
                "self_contradiction", "statement contradicts itself (e.g., 'I cannot... but I can')"
            )

        if solution.confidence < self.min_confidence:
            return False, RedFlagRejection("low_confidence", f"confidence {solution.confidence:.2f} below threshold")

        return True, None

    # Internal helpers -------------------------------------------------
    def _has_repetition(self, text: str) -> bool:
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        counts = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= self.repetition_threshold:
                return True

        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < 4:
            return False
        window = []
        seen = {}
        for token in tokens:
            window.append(token)
            if len(window) > 4:
                window.pop(0)
            if len(window) < 2:
                continue
            key = tuple(window)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= self.repetition_threshold:
                return True
        return False
