"""Default domain adapter with basic text verification.

This adapter provides minimal verification suitable for general-purpose tasks.
"""
from typing import Any, Dict, List, Tuple

from .base import BaseDomainAdapter


class DefaultAdapter(BaseDomainAdapter):
    """Generic adapter with basic text verification.

    Use this for tasks that don't require domain-specific validation.
    """

    name = "default"

    def verify_solution(self, solution: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify a solution with basic checks.

        Args:
            solution: The solution text to verify.
            context: Additional context (unused for default).

        Returns:
            Tuple of (is_valid, reason).
        """
        if not solution:
            return False, "solution is empty"

        if not solution.strip():
            return False, "solution contains only whitespace"

        if len(solution.strip()) < 3:
            return False, "solution too short (< 3 characters)"

        return True, "ok"

    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """Extract basic evidence from text.

        Args:
            text: Solution text to extract from.

        Returns:
            Dictionary with raw text field.
        """
        return {
            "raw_text": text.strip(),
            "length": len(text),
        }

    def compose_results(self, results: List[str], compose_fn: str) -> str:
        """Compose results by concatenation.

        Args:
            results: List of sub-result strings.
            compose_fn: Instructions for combination.

        Returns:
            Composed result string.
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0].strip()

        parts = [
            f"{compose_fn.strip()}",
            "",
        ]

        for i, result in enumerate(results):
            parts.append(f"Part {i + 1}:")
            parts.append(result.strip())
            parts.append("")

        return "\n".join(parts).strip()

    def get_calibration_problems(self) -> List[str]:
        """Return simple calibration problems.

        Returns:
            List of basic problems for calibration.
        """
        return [
            "What is 2 + 2?",
            "What is the capital of France?",
            "What color is the sky on a clear day?",
        ]
