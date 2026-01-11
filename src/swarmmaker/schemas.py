"""Data contracts and configuration for the MAKER SwarmMaker runtime.

This module consolidates all schemas, configuration, and settings.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Protocol, Tuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Environment / Settings (merged from config.py)
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def _init_env() -> None:
    """Load .env file at module import if python-dotenv is available."""
    if _load_dotenv is None:
        return
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        _load_dotenv(dotenv_path=env_path, override=False)


_init_env()


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Return canonical JSON for deterministic signatures and logs."""

    def default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)


def signature_hash(text: str) -> str:
    """Return a short hash for deduplication/comparison."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StructuredMode(str, Enum):
    """Supported response-format modes for LLM providers."""

    json_schema = "json_schema"
    json_object = "json_object"


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class ModelRoles(BaseModel):
    """Role-based model configuration.

    Instead of hardcoding models per component, we define two roles:
    - reasoning: For decomposition, composition, completeness checking (higher capability)
    - execution: For atomic solving (can be smaller/faster)
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for decomposition, composition, and completeness checking.",
    )
    execution: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for atomic solving. Can be a smaller/faster model.",
    )


class ThresholdConfig(BaseModel):
    """Centralized threshold configuration.

    All magic numbers and thresholds are defined here for easy tuning.
    Based on the MAKER paper recommendations where applicable.
    """

    model_config = ConfigDict(extra="forbid")

    # Voting thresholds (paper-aligned)
    min_samples_for_confidence: int = Field(
        default=3,
        ge=1,
        description="Minimum samples before declaring confident consensus.",
    )
    temperature_first_vote: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for first voting sample (paper recommends 0).",
    )
    temperature_subsequent: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for subsequent voting samples (paper recommends 0.1).",
    )

    # Red-flag thresholds (paper: >700 tokens correlates with ~90% error)
    max_solution_tokens: int = Field(
        default=750,
        ge=100,
        description="Max estimated tokens for atomic solutions. Paper shows >700 = 90% error.",
    )
    max_solution_chars: int = Field(
        default=3000,
        ge=400,
        description="Max characters for atomic solutions (~4 chars per token backup).",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to accept a solution.",
    )
    repetition_threshold: int = Field(
        default=3,
        ge=2,
        description="Number of repetitions to trigger rejection.",
    )

    # Output token limits
    max_output_tokens: int = Field(
        default=4096,
        ge=256,
        description="Maximum output tokens for LLM responses (increase for code generation).",
    )

    # Verification thresholds
    sympy_tolerance: float = Field(
        default=1e-5,
        gt=0,
        description="Tolerance for sympy numerical verification.",
    )

    # State management
    max_facts: int = Field(
        default=5,
        ge=1,
        description="Maximum facts to keep in state (FIFO eviction).",
    )
    max_notes: int = Field(
        default=20,
        ge=5,
        description="Maximum notes to keep in state (prevents memory bloat).",
    )
    max_work_shown_chars: int = Field(
        default=1200,
        ge=100,
        description="Truncate work_shown to this length.",
    )
    state_summary_char_limit: int = Field(
        default=1200,
        ge=100,
        description="Limit for state summaries in prompts.",
    )


class CalibrationConfig(BaseModel):
    """Configuration for pre-run calibration phase.

    The MAKER paper recommends estimating per-step success probability (p)
    before running, then calculating optimal ahead-by-k value.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether to run calibration phase before main execution.",
    )
    samples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of calibration samples to run.",
    )
    target_error_rate: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="Target error rate for calculating optimal k.",
    )
    fallback_p: float = Field(
        default=0.7,
        gt=0,
        lt=1,
        description="Assumed success probability if calibration is skipped.",
    )
    max_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum ahead-by-k value to use.",
    )


class SwarmConfig(BaseModel):
    """Runtime configuration for the MAKER orchestrator.

    This is the main configuration object passed throughout the system.
    """

    model_config = ConfigDict(extra="forbid")

    # Model configuration (role-based)
    models: ModelRoles = Field(default_factory=ModelRoles)

    # Threshold configuration (centralized)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)

    # Calibration configuration
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    # Core algorithm parameters
    batch_size: int = Field(default=4, ge=1, description="Samples per round for voting.")
    ahead_by: int = Field(default=2, ge=1, description="Votes required beyond runner-up.")
    max_rounds: int = Field(default=5, ge=1, description="Max sampling rounds per stage.")
    max_depth: int = Field(default=6, ge=1, description="Maximum recursion depth.")
    max_stagnant_rounds: int = Field(default=2, ge=1, description="Rounds without progress before escalation.")

    # Resource limits
    max_total_tokens: int = Field(default=50_000, ge=1000, description="Token budget for all LLM calls.")
    timeout_seconds: int = Field(default=60, ge=10, description="Timeout per LLM call.")
    timeout_total_seconds: int = Field(default=600, ge=60, description="Total orchestrator timeout.")

    # Execution control
    dry_run: bool = Field(default=False, description="Return mock results without API calls.")
    log_dir: Path = Field(default_factory=lambda: Path("runs"), description="Directory for artifacts.")
    structured_mode: StructuredMode = Field(
        default=StructuredMode.json_schema,
        description="LLM response format enforcement mode.",
    )

    # Domain adapter
    domain: str = Field(default="default", description="Domain adapter to use (default, math, etc.).")

    def get_model(self, role: Literal["reasoning", "execution"]) -> str:
        """Get model for a specific role."""
        return getattr(self.models, role)


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


class DecompositionProposal(BaseModel):
    """LLM-proposed decomposition of a problem into two subproblems."""

    model_config = ConfigDict(extra="forbid")

    subproblem_a: str = Field(min_length=1, description="First subproblem to solve.")
    subproblem_b: str = Field(min_length=1, description="Second subproblem to solve.")
    compose_fn: str = Field(min_length=1, description="Instructions to combine results.")
    is_atomic: bool = Field(
        description="True when the problem cannot be decomposed further.",
    )
    rationale: str = Field(min_length=1, description="Why this decomposition is appropriate.")

    @cached_property
    def signature(self) -> str:
        return canonical_json({
            "subproblem_a": self.subproblem_a.strip(),
            "subproblem_b": self.subproblem_b.strip(),
            "compose_fn": self.compose_fn.strip(),
            "is_atomic": self.is_atomic,
        })


class AtomicSolution(BaseModel):
    """Atomic solver output containing the proposed answer."""

    model_config = ConfigDict(extra="forbid")

    solution: str = Field(min_length=1, description="Candidate solution string.")
    confidence: float = Field(ge=0.0, le=1.0)
    work_shown: str = Field(min_length=1, description="Audit trail for the answer.")

    @cached_property
    def signature(self) -> str:
        return canonical_json({"solution": self.solution.strip()})


class ProblemFact(BaseModel):
    """Compact fact captured from an accepted atomic solution."""

    model_config = ConfigDict(extra="forbid")

    problem: str
    solution: str
    work_shown: str
    confidence: float = Field(ge=0.0, le=1.0)
    depth: int = 0


class SupportPoint(BaseModel):
    """Structured support point describing a computed result."""

    model_config = ConfigDict(extra="forbid")

    label: str
    x: float = Field(default=0.0, description="X coordinate value.")
    y: float = Field(default=0.0, description="Y coordinate value.")
    statement: str = Field(default="", description="Description of this point.")


class FinalSupport(BaseModel):
    """Optional support payload carried with the final answer."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(default="", description="Brief summary of the support evidence.")
    equations: List[str] = Field(default_factory=list)
    points: List[SupportPoint] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """Final answer payload returned by the composer stage."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Direct response to the original task.")
    confidence: float = Field(ge=0.0, le=1.0)
    support: Optional[FinalSupport] = Field(default=None, description="Optional supporting evidence.")


class RequirementStatus(BaseModel):
    """Status of a single requirement from the task."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(description="The requirement extracted from the task.")
    status: Literal["ADDRESSED", "MISSING"] = Field(description="Whether this requirement is addressed.")
    reason: str = Field(description="Why the requirement is addressed or missing.")


class CompletenessResult(BaseModel):
    """Result of checking if all task requirements are addressed."""

    model_config = ConfigDict(extra="forbid")

    requirements: List[RequirementStatus] = Field(default_factory=list)
    complete: bool = Field(description="True if all requirements are addressed.")
    missing_work: List[str] = Field(
        default_factory=list,
        description="Specific tasks to perform to fill gaps, if any.",
    )


class TaskState(BaseModel):
    """Minimal state snapshot shared with the orchestrator."""

    task: str
    decomposition_tree: Dict[str, Any] = Field(default_factory=dict)
    solved_subproblems: Dict[str, str] = Field(default_factory=dict)
    facts: Dict[str, ProblemFact] = Field(default_factory=dict)
    current_problem: str
    depth: int = 0
    notes: List[str] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    progress_version: int = 0
    progress_hash: str = ""

    def add_note(self, note: str, max_notes: int = 20) -> None:
        """Add a note with bounded list size."""
        self.notes.append(note)
        if len(self.notes) > max_notes:
            self.notes = self.notes[-max_notes:]


class StepTrace(BaseModel):
    """History entry captured for each decomposition or atomic step."""

    step_id: int
    kind: Literal["decomposition", "atomic"]
    problem: str
    depth: int
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    chosen: Optional[Dict[str, Any]] = None
    votes: Dict[str, int] = Field(default_factory=dict)
    notes: Optional[str] = None
    rejections: List[Dict[str, Any]] = Field(default_factory=list)


class RunStats(BaseModel):
    elapsed_s: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    consensus_votes: int = 0
    aborted_reason: Optional[str] = None


class RunArtifacts(BaseModel):
    result_md_path: Optional[Path] = None
    result_json_path: Optional[Path] = None
    events_path: Optional[Path] = None
    langsmith_run_url: Optional[str] = None


class RunResult(BaseModel):
    task: str
    final_answer: Optional[str]
    final_payload: Optional[FinalAnswer] = None
    steps: List[StepTrace]
    stats: RunStats
    artifacts: RunArtifacts
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Runtime Types
# ---------------------------------------------------------------------------


@dataclass
class AgentCallMeta:
    """Metadata for tagging LLM calls."""

    agent: str
    stage: str
    step_id: int
    voter_id: Optional[int] = None


T = TypeVar("T")


@dataclass
class StructuredParseResult(Generic[T]):
    """Return type for structured LLM responses."""

    content: T
    raw_text: str


class SchemaValidationError(RuntimeError):
    """Raised when provider violates the agreed schema."""

    def __init__(self, *, agent: str, stage: str, payload: str, error: Exception):
        super().__init__("Provider did not honor structured output")
        self.agent = agent
        self.stage = stage
        self.payload = payload
        self.error = error


# ---------------------------------------------------------------------------
# Plugin Protocols
# ---------------------------------------------------------------------------


class RedFlagPattern(Protocol):
    """Protocol for pluggable red-flag detection patterns."""

    name: str

    def check(self, text: str) -> Optional[str]:
        """Check text for this pattern. Return rejection reason or None if ok."""
        ...


class DomainAdapter(Protocol):
    """Protocol for domain-specific verification and composition."""

    name: str

    def verify_solution(self, solution: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify a solution is valid for this domain. Returns (is_valid, reason)."""
        ...

    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """Extract structured evidence from solution text."""
        ...

    def compose_results(self, results: List[str], compose_fn: str) -> str:
        """Compose multiple sub-results into a final answer."""
        ...

    def get_red_flag_patterns(self) -> List[RedFlagPattern]:
        """Return domain-specific red flag patterns."""
        ...

    def get_calibration_problems(self) -> List[str]:
        """Return calibration problems for this domain."""
        ...
