"""Domain-specific typed state for SwarmMaker.

This module replaces free-text notes with structured, domain-specific state
that enables progress tracking, stagnation detection, and domain-specific validation.
"""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel, Field


class DomainState(BaseModel, ABC):
    """Base class for domain-specific state.

    All domain states must implement methods for merging actions,
    hashing for stagnation detection, and progress scoring.
    """

    @abstractmethod
    def merge(self, action: Any) -> "DomainState":
        """Return new state with action applied.

        Args:
            action: Action object to merge into state

        Returns:
            New DomainState instance with action's effects applied
        """
        pass

    @abstractmethod
    def hash_key(self) -> str:
        """Generate canonical hash for stagnation detection.

        Returns:
            Hex digest of canonical JSON representation
        """
        pass

    @abstractmethod
    def progress_score(self, prev: "DomainState") -> float:
        """Measure progress from previous state to current.

        Args:
            prev: Previous state snapshot

        Returns:
            Progress score: 0.0 = no progress, 1.0 = major progress
        """
        pass


class MathState(DomainState):
    """State for mathematical problem solving.

    Tracks equations, simplifications, solutions, and intermediate results
    to enable progress measurement and stagnation detection.
    """

    equations: List[str] = Field(default_factory=list, description="All equations encountered")
    simplifications: List[Tuple[str, str]] = Field(
        default_factory=list, description="Transformations (from, to)"
    )
    solutions: Dict[str, Any] = Field(default_factory=dict, description="Variable solutions")
    intermediate_results: List[str] = Field(
        default_factory=list, description="Intermediate findings"
    )

    def merge(self, action: Any) -> "MathState":
        """Apply action to math state by extracting mathematical content.

        Args:
            action: Action with args.content containing math work

        Returns:
            New MathState with extracted content added
        """
        new_state = self.model_copy(deep=True)

        content = action.args.get("content", "") if hasattr(action, "args") else ""
        if not content:
            return new_state

        # Extract equations (lines with = signs, excluding comparisons)
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            # Match mathematical equations (not ==, !=, <=, >=)
            if re.search(r'(?<![<>!=])=(?!=)', line_stripped) and line_stripped:
                # Clean up equation
                equation = line_stripped
                # Remove trailing punctuation
                equation = re.sub(r'[,;.]+$', '', equation)
                if equation and equation not in new_state.equations:
                    new_state.equations.append(equation)

        # Extract solutions (x = numeric value or simple expression)
        solution_pattern = r'([a-z])\s*=\s*([-+]?\d+\.?\d*(?:/\d+)?)'
        for match in re.finditer(solution_pattern, content, re.IGNORECASE):
            var, value = match.groups()
            # Try to parse as number
            try:
                if '/' in value:
                    parts = value.split('/')
                    numeric_val = float(parts[0]) / float(parts[1])
                else:
                    numeric_val = float(value)
                new_state.solutions[var.lower()] = numeric_val
            except ValueError:
                # Store as string if can't parse
                new_state.solutions[var.lower()] = value

        # Add to intermediate results (non-empty lines)
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in new_state.intermediate_results:
                new_state.intermediate_results.append(line_stripped)

        return new_state

    def hash_key(self) -> str:
        """Hash based on equations and solutions."""
        key_data = {
            "equations": sorted(self.equations),
            "solutions": dict(sorted(self.solutions.items())),
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def progress_score(self, prev: "MathState") -> float:
        """Score based on new equations and solutions found."""
        new_equations = len(self.equations) - len(prev.equations)
        new_solutions = len(self.solutions) - len(prev.solutions)

        if new_solutions > 0:
            return 1.0  # Found solution = max progress
        elif new_equations > 0:
            return 0.7  # New equation = good progress
        elif self.hash_key() != prev.hash_key():
            return 0.3  # State changed but no structural progress
        else:
            return 0.0  # No progress


class CodeState(DomainState):
    """State for code generation and modification tasks.

    Tracks files, functions, tests, and errors to measure code development progress.
    """

    files: Dict[str, str] = Field(
        default_factory=dict, description="filename -> content mapping"
    )
    functions: List[str] = Field(default_factory=list, description="Function signatures")
    tests: List[str] = Field(default_factory=list, description="Test names/descriptions")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    def merge(self, action: Any) -> "CodeState":
        """Extract code artifacts from action content."""
        new_state = self.model_copy(deep=True)

        content = action.args.get("content", "") if hasattr(action, "args") else ""
        if not content:
            return new_state

        # Extract function definitions (simple heuristic)
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            if func_name not in new_state.functions:
                new_state.functions.append(func_name)

        # Extract test names
        test_pattern = r'(test_[a-zA-Z0-9_]+)'
        for match in re.finditer(test_pattern, content):
            test_name = match.group(1)
            if test_name not in new_state.tests:
                new_state.tests.append(test_name)

        # Extract file references (filename.ext pattern)
        file_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-z]{1,4})'
        for match in re.finditer(file_pattern, content):
            filename = match.group(1)
            if filename not in new_state.files:
                # Store reference (actual content would come from file operations)
                new_state.files[filename] = ""

        return new_state

    def hash_key(self) -> str:
        """Hash based on files, functions, and tests."""
        key_data = {
            "files": sorted(self.files.keys()),
            "functions": sorted(self.functions),
            "tests": sorted(self.tests),
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def progress_score(self, prev: "CodeState") -> float:
        """Score based on new code artifacts."""
        new_functions = len(self.functions) - len(prev.functions)
        new_tests = len(self.tests) - len(prev.tests)
        new_files = len(self.files) - len(prev.files)

        total_new = new_functions + new_tests + new_files

        if total_new >= 3:
            return 1.0  # Multiple new artifacts = major progress
        elif total_new > 0:
            return 0.6  # Some new artifacts = moderate progress
        elif self.hash_key() != prev.hash_key():
            return 0.3  # State changed minimally
        else:
            return 0.0  # No progress


class CreativeState(DomainState):
    """State for creative tasks like writing and brainstorming.

    Tracks ideas, themes, and draft sections to measure creative progress.
    """

    ideas: List[str] = Field(default_factory=list, description="Generated ideas")
    themes: List[str] = Field(default_factory=list, description="Identified themes")
    draft_sections: Dict[str, str] = Field(
        default_factory=dict, description="section_name -> content"
    )

    def merge(self, action: Any) -> "CreativeState":
        """Extract creative content from action."""
        new_state = self.model_copy(deep=True)

        content = action.args.get("content", "") if hasattr(action, "args") else ""
        if not content:
            return new_state

        # Extract bullet points as ideas
        bullet_pattern = r'^\s*[-*•]\s*(.+)$'
        for line in content.split('\n'):
            match = re.match(bullet_pattern, line)
            if match:
                idea = match.group(1).strip()
                if idea and idea not in new_state.ideas:
                    new_state.ideas.append(idea)

        # Extract section headers as themes/topics
        header_pattern = r'^#+\s*(.+)$'
        for line in content.split('\n'):
            match = re.match(header_pattern, line)
            if match:
                theme = match.group(1).strip()
                if theme and theme not in new_state.themes:
                    new_state.themes.append(theme)

        return new_state

    def hash_key(self) -> str:
        """Hash based on ideas and themes."""
        key_data = {
            "ideas": sorted(self.ideas),
            "themes": sorted(self.themes),
            "draft_sections": dict(sorted(self.draft_sections.items())),
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def progress_score(self, prev: "CreativeState") -> float:
        """Score based on new ideas and themes."""
        new_ideas = len(self.ideas) - len(prev.ideas)
        new_themes = len(self.themes) - len(prev.themes)
        new_sections = len(self.draft_sections) - len(prev.draft_sections)

        total_new = new_ideas + new_themes + new_sections

        if total_new >= 5:
            return 1.0  # Many new ideas = major progress
        elif total_new > 0:
            return 0.6  # Some new ideas = moderate progress
        elif self.hash_key() != prev.hash_key():
            return 0.3  # State changed minimally
        else:
            return 0.0  # No progress


class GenericState(DomainState):
    """Fallback state for tasks without specific domain type.

    Uses simple text accumulation similar to the old notes system,
    but with structured progress tracking.
    """

    content: List[str] = Field(default_factory=list, description="Accumulated content")

    def merge(self, action: Any) -> "GenericState":
        """Add action content to accumulated content."""
        new_state = self.model_copy(deep=True)

        content = action.args.get("content", "") if hasattr(action, "args") else ""
        if content and content not in new_state.content:
            new_state.content.append(content)

        return new_state

    def hash_key(self) -> str:
        """Hash all content."""
        return hashlib.sha256(
            json.dumps(sorted(self.content), sort_keys=True).encode()
        ).hexdigest()

    def progress_score(self, prev: "GenericState") -> float:
        """Score based on new content added."""
        new_content = len(self.content) - len(prev.content)

        if new_content > 0:
            return 0.5  # New content = moderate progress
        elif self.hash_key() != prev.hash_key():
            return 0.2  # Hash changed minimally
        else:
            return 0.0  # No progress


class DomainStateFactory:
    """Factory for creating appropriate DomainState based on task category."""

    @staticmethod
    def create(category: str) -> DomainState:
        """Create domain state instance for given category.

        Args:
            category: Task category (MATH, CODE, CREATIVE, etc.)

        Returns:
            Appropriate DomainState instance
        """
        category_upper = category.upper() if isinstance(category, str) else "GENERIC"

        if category_upper == "MATH":
            return MathState()
        elif category_upper == "CODE":
            return CodeState()
        elif category_upper == "CREATIVE":
            return CreativeState()
        else:
            return GenericState()
