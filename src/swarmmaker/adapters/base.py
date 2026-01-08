"""Base class for domain adapters.

Domain adapters provide pluggable verification, evidence extraction,
and composition capabilities for different problem domains.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseDomainAdapter(ABC):
    """Abstract base class for domain-specific adapters.

    Subclass this to create adapters for specific domains like math,
    code, writing, etc.
    """

    name: str = "base"

    @abstractmethod
    def verify_solution(self, solution: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify a solution is valid for this domain.

        Args:
            solution: The solution text to verify.
            context: Additional context (task, equations, etc.)

        Returns:
            Tuple of (is_valid, reason).
        """
        pass

    @abstractmethod
    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """Extract structured evidence from solution text.

        Args:
            text: Solution text to extract from.

        Returns:
            Dictionary with domain-specific evidence fields.
        """
        pass

    @abstractmethod
    def compose_results(self, results: List[str], compose_fn: str) -> str:
        """Compose multiple sub-results into a final answer.

        Args:
            results: List of sub-result strings.
            compose_fn: Instructions for how to combine results.

        Returns:
            Composed result string.
        """
        pass

    def get_red_flag_patterns(self) -> List[Any]:
        """Return domain-specific red flag patterns.

        Override to add patterns specific to this domain.

        Returns:
            List of RedFlagPattern instances.
        """
        return []

    def get_calibration_problems(self) -> List[str]:
        """Return calibration problems for estimating success rate.

        Override to provide domain-specific calibration tasks.

        Returns:
            List of simple problem strings for calibration.
        """
        return []
