"""Domain-specific validation for mathematical tasks.

This module provides math-specific validation that goes beyond string heuristics,
checking for actual mathematical content and structural validity.
"""

import re
from typing import Tuple

from .domain_state import MathState


class MathValidator:
    """Domain-specific validation for math tasks."""

    def validate_action(self, action, math_state: MathState) -> Tuple[bool, str]:
        """Validate math action for correctness and completeness.

        Args:
            action: Action object with args.content
            math_state: Current MathState

        Returns:
            Tuple of (valid, reason). If not valid, reason explains the issue.
        """
        content = action.args.get("content", "") if hasattr(action, "args") else ""

        if not content or not content.strip():
            return False, "Math action must contain non-empty content"

        # RULE 1: Must contain actual math symbols or numbers
        has_math_symbols = any(c in content for c in "=+-*/^()")
        has_numbers = bool(re.search(r'\d', content))

        if not (has_math_symbols or has_numbers):
            return False, "Math action must contain equations, operations, or numeric values"

        # RULE 2: For FINAL actions, must contain solution
        if hasattr(action, "action_type") and action.action_type == "FINAL":
            # Check for variable solution pattern (x = value)
            solution_pattern = r'[a-z]\s*=\s*[-+]?\d+\.?\d*'
            if not re.search(solution_pattern, content, re.IGNORECASE):
                # Also accept verbal statement of solution
                verbal_patterns = [
                    r'(solution|answer|result)\s*(?:is|:)\s*[-+]?\d+',
                    r'[a-z]\s+equals?\s+[-+]?\d+',
                ]
                has_verbal = any(re.search(p, content, re.IGNORECASE) for p in verbal_patterns)

                if not has_verbal:
                    return False, "Final math answer must include solved value (e.g., 'x = 5' or 'solution is 42')"

        # RULE 3: Check for balanced parentheses
        try:
            self._check_balanced_parentheses(content)
        except ValueError as e:
            return False, f"Invalid mathematical expression: {e}"

        # RULE 4: Reject if content is just restating the problem
        # (This is a heuristic - we check if content looks like a problem statement)
        problem_indicators = ["solve", "find", "calculate", "determine", "what is"]
        lines = content.lower().split('\n')
        problem_line_count = sum(
            1 for line in lines if any(indicator in line for indicator in problem_indicators)
        )

        # If majority of non-empty lines are problem statements, it's likely not actual work
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines and problem_line_count / len(non_empty_lines) > 0.7:
            return False, "Math action appears to restate problem without showing work"

        return True, "valid"

    def _check_balanced_parentheses(self, content: str) -> None:
        """Check for balanced parentheses in mathematical expressions.

        Args:
            content: Content to check

        Raises:
            ValueError: If parentheses are unbalanced
        """
        # Count different bracket types
        open_count = content.count('(')
        close_count = content.count(')')

        if open_count != close_count:
            raise ValueError(f"Unbalanced parentheses: {open_count} open, {close_count} close")

        # Check proper nesting (stack-based)
        stack = []
        for char in content:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    raise ValueError("Closing parenthesis without matching opening parenthesis")
                stack.pop()

        if stack:
            raise ValueError("Unclosed opening parenthesis")
