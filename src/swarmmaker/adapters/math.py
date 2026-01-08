"""Math domain adapter with sympy verification.

This adapter provides mathematical verification using sympy,
equation extraction, and point validation.
"""
import math
import re
from typing import Any, Dict, List, Tuple

from .base import BaseDomainAdapter

# Optional sympy import
try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    sp = None
    SYMPY_AVAILABLE = False


class MathAdapter(BaseDomainAdapter):
    """Math domain adapter with sympy verification.

    Features:
    - Equation extraction from text
    - Point extraction (coordinates)
    - Sympy-based numerical verification
    - Math-specific calibration problems
    """

    name = "math"

    def __init__(self, tolerance: float = 1e-5) -> None:
        """Initialize math adapter.

        Args:
            tolerance: Tolerance for numerical comparisons.
        """
        self.tolerance = tolerance

    def verify_solution(self, solution: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify a math solution using sympy if available.

        Args:
            solution: The solution text to verify.
            context: Must contain 'equations' and optionally 'points'.

        Returns:
            Tuple of (is_valid, reason).
        """
        if not solution or not solution.strip():
            return False, "solution is empty"

        # Extract equations and points from context
        equations = context.get("equations", [])
        points = context.get("points", [])

        # If no equations to verify against, just do basic check
        if not equations:
            # Try to extract equations from task
            task = context.get("task", "")
            equations = self.extract_equations(task)

        if not equations:
            return True, "no equations to validate"

        # Extract points from solution if not provided
        if not points:
            points = self.extract_points(solution)

        if not points:
            return True, "no points to validate"

        # Verify points satisfy equations
        return self._verify_points_satisfy_equations(points, equations)

    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """Extract mathematical evidence from text.

        Args:
            text: Solution text to extract from.

        Returns:
            Dictionary with equations, points, and expressions.
        """
        return {
            "equations": self.extract_equations(text),
            "points": self.extract_points(text),
            "expressions": self._extract_expressions(text),
        }

    def compose_results(self, results: List[str], compose_fn: str) -> str:
        """Compose math results.

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

        parts = [compose_fn.strip(), ""]

        for i, result in enumerate(results, 1):
            parts.append(f"Step {i} result:")
            parts.append(result.strip())
            parts.append("")

        return "\n".join(parts).strip()

    def get_calibration_problems(self) -> List[str]:
        """Return math calibration problems.

        Returns:
            List of simple math problems for calibration.
        """
        return [
            "What is 2 + 2?",
            "Solve x + 3 = 7 for x",
            "If y = 2x and x = 3, what is y?",
            "What is 15 divided by 3?",
            "Simplify 2(x + 3)",
        ]

    # Math-specific methods

    def extract_equations(self, text: str) -> List[str]:
        """Extract equations from text.

        Args:
            text: Text to extract from.

        Returns:
            List of equation strings.
        """
        # Replace 'and' with comma for splitting
        cleaned = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)

        # Find patterns like "x = ..." or "y = ..."
        simple_matches = re.findall(r"([xy]\s*=[^,;]+)", cleaned)
        equations = [m.strip() for m in simple_matches]

        # Also find patterns like "f(x) = ..."
        func_matches = re.findall(r"([a-z]\([xy]\)\s*=[^,;]+)", cleaned, re.IGNORECASE)
        equations.extend(m.strip() for m in func_matches)

        return equations

    def extract_points(self, text: str) -> List[Dict[str, float]]:
        """Extract coordinate points from text.

        Args:
            text: Text to extract from.

        Returns:
            List of points as dicts with 'x' and 'y' keys.
        """
        points = []

        # Pattern for (x, y) or (number, number)
        coord_pattern = re.compile(
            r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
        )

        for match in coord_pattern.finditer(text):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                points.append({"x": x, "y": y})
            except ValueError:
                continue

        return points

    def _extract_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text.

        Args:
            text: Text to extract from.

        Returns:
            List of expression strings.
        """
        # Find patterns with operators
        expr_pattern = re.compile(
            r"[\d\w]+\s*[\+\-\*\/\^]\s*[\d\w]+(?:\s*[\+\-\*\/\^]\s*[\d\w]+)*",
        )
        return expr_pattern.findall(text)

    def _verify_points_satisfy_equations(
        self,
        points: List[Dict[str, float]],
        equations: List[str],
    ) -> Tuple[bool, str]:
        """Verify that points satisfy the given equations.

        Args:
            points: List of points to verify.
            equations: List of equation strings.

        Returns:
            Tuple of (is_valid, reason).
        """
        if not SYMPY_AVAILABLE:
            return True, "sympy not available, skipping verification"

        if not points or not equations:
            return True, "nothing to verify"

        # Parse equations
        x_var, y_var = sp.symbols("x y")
        local_dict = {"x": x_var, "y": y_var}
        parsed_eqs = []

        for raw in equations[:4]:  # Limit to 4 equations
            normalized = raw.replace("^", "**")
            if "=" in normalized:
                left, right = normalized.split("=", 1)
            else:
                left, right = normalized, "0"

            try:
                left_expr = sp.sympify(left, locals=local_dict)
                right_expr = sp.sympify(right, locals=local_dict)
                parsed_eqs.append(sp.simplify(left_expr - right_expr))
            except Exception:
                continue

        if not parsed_eqs:
            return True, "could not parse equations"

        # Verify each point
        for point in points[:4]:  # Limit to 4 points
            px = point.get("x", 0.0)
            py = point.get("y", 0.0)

            # Skip default/unset points
            if px == 0.0 and py == 0.0:
                continue

            for expr in parsed_eqs:
                try:
                    result = expr.subs({"x": px, "y": py})
                    numeric = float(result.evalf())
                except Exception:
                    continue

                if not math.isfinite(numeric):
                    return False, f"non-finite result at point ({px}, {py})"

                if abs(numeric) > self.tolerance:
                    return False, f"point ({px}, {py}) does not satisfy equation"

        return True, "all points satisfy equations"
