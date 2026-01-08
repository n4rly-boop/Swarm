"""Calibration module for estimating success probability.

Based on the MAKER paper's recommendation to estimate per-step success
probability (p) before running, then calculate optimal ahead-by-k value.
"""
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .schemas import CalibrationConfig, SwarmConfig


@dataclass
class CalibrationResult:
    """Result of the calibration phase."""

    estimated_p: float  # Estimated per-step success probability
    optimal_k: int  # Calculated optimal ahead-by-k value
    samples_used: int  # Number of calibration samples run
    correct_count: int  # Number of correct samples
    elapsed_s: float  # Time spent on calibration


class Calibrator:
    """Estimates per-step success probability before full run.

    The MAKER paper shows that:
    - p_full = (1 + ((1-p)/p)^k)^(-s/m)
    - Where p is per-step success rate, k is ahead-by value, s is steps, m is steps per subtask

    This calibrator:
    1. Runs simple calibration problems
    2. Measures success rate
    3. Calculates optimal k for target error rate
    """

    def __init__(self, config: "CalibrationConfig") -> None:
        """Initialize calibrator.

        Args:
            config: Calibration configuration.
        """
        self.config = config

    def calculate_optimal_k(
        self,
        p: float,
        target_error_rate: Optional[float] = None,
    ) -> int:
        """Calculate optimal ahead-by-k value for given success probability.

        Uses the MAKER paper's formula to find minimum k such that
        the expected error rate is below the target.

        Args:
            p: Estimated per-step success probability (0 < p < 1).
            target_error_rate: Target error rate. Uses config default if None.

        Returns:
            Optimal k value.
        """
        target = target_error_rate or self.config.target_error_rate

        # Handle edge cases
        if p <= 0.5:
            # Low accuracy - need maximum consensus
            return self.config.max_k

        if p >= 0.99:
            # Very high accuracy - minimal consensus needed
            return 1

        # Calculate k using the formula: error_rate ≈ ((1-p)/p)^k
        # We want ((1-p)/p)^k < target_error_rate
        # So k > log(target_error_rate) / log((1-p)/p)

        ratio = (1 - p) / p
        if ratio <= 0:
            return 1

        try:
            log_ratio = math.log(ratio)
            log_target = math.log(target)

            if log_ratio >= 0:
                # p <= 0.5, use max k
                return self.config.max_k

            k = math.ceil(log_target / log_ratio)
            return max(1, min(k, self.config.max_k))

        except (ValueError, ZeroDivisionError):
            return self.config.max_k

    def estimate_p_from_samples(
        self,
        correct_count: int,
        total_count: int,
        prior_p: float = 0.7,
    ) -> float:
        """Estimate success probability from sample counts.

        Uses Bayesian estimation with a prior to avoid extreme estimates
        from small samples.

        Args:
            correct_count: Number of correct samples.
            total_count: Total number of samples.
            prior_p: Prior probability estimate.

        Returns:
            Estimated success probability.
        """
        if total_count == 0:
            return prior_p

        # Simple Bayesian estimate with pseudo-counts
        # Equivalent to Beta(prior_strength * prior_p, prior_strength * (1-prior_p))
        prior_strength = 2.0  # Weight given to prior
        pseudo_correct = prior_strength * prior_p
        pseudo_total = prior_strength

        return (correct_count + pseudo_correct) / (total_count + pseudo_total)

    def create_result(
        self,
        correct_count: int,
        total_count: int,
        elapsed_s: float,
    ) -> CalibrationResult:
        """Create calibration result from sample data.

        Args:
            correct_count: Number of correct samples.
            total_count: Total number of samples.
            elapsed_s: Time spent on calibration.

        Returns:
            CalibrationResult with estimated p and optimal k.
        """
        estimated_p = self.estimate_p_from_samples(
            correct_count,
            total_count,
            self.config.fallback_p,
        )

        optimal_k = self.calculate_optimal_k(estimated_p)

        return CalibrationResult(
            estimated_p=estimated_p,
            optimal_k=optimal_k,
            samples_used=total_count,
            correct_count=correct_count,
            elapsed_s=elapsed_s,
        )

    def get_default_result(self) -> CalibrationResult:
        """Get default result when calibration is skipped.

        Returns:
            CalibrationResult using fallback values.
        """
        return CalibrationResult(
            estimated_p=self.config.fallback_p,
            optimal_k=self.calculate_optimal_k(self.config.fallback_p),
            samples_used=0,
            correct_count=0,
            elapsed_s=0.0,
        )
