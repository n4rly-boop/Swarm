# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project: MAKER

**MAKER** (Maximal Agentic decomposition + first-to-Ahead-by-K Error correction + Red-flagging) is a framework for executing long-horizon tasks with near-zero error rate by converting them into many tiny, independently correctable steps.

**Core Principle**: Never trust a single long reasoning chain. Trust *many small steps*, each protected by **sampling, filtering, and voting**.

---

## Commands

### Setup and Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running
```bash
# Show options
python -m swarmmaker --help

# Execute a task
python -m swarmmaker "your task here"

# Dry run (no API calls)
python -m swarmmaker "task" --dry-run

# Custom configuration with role-based models
python -m swarmmaker "task" \
  --model-reasoning anthropic/claude-sonnet-4 \
  --model-execution qwen/qwen2.5-coder-7b-instruct \
  --batch-size 5 \
  --ahead-by 2 \
  --max-rounds 10

# With calibration phase (estimates optimal K)
python -m swarmmaker "task" --calibrate

# Math domain (sympy verification)
python -m swarmmaker "solve x^2 = 4" --domain math
```

### Environment Variables
```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional
export LANGSMITH_API_KEY=...                              # optional tracing
export LANGCHAIN_PROJECT="maker"
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-reasoning, -r` | Decomposition, composition, completeness | `claude-sonnet-4` |
| `--model-execution, -e` | Atomic solving | `claude-sonnet-4` |
| `--batch-size` | Samples per round | 4 |
| `--ahead-by` | Votes required beyond runner-up | 2 |
| `--max-rounds` | Max sampling rounds | 5 |
| `--max-depth` | Maximum recursion depth | 6 |
| `--max-total-tokens` | Token budget | 50,000 |
| `--timeout-seconds` | Per-call timeout | 60s |
| `--timeout-total` | Total orchestrator timeout | 600s |
| `--calibrate` | Enable pre-run calibration phase | False |
| `--domain` | Adapter: "default", "math" | "default" |
| `--structured-mode` | json_schema or json_object | json_schema |
| `--dry-run` | Mock results without API calls | False |
| `--log-dir` | Output directory | `runs/<timestamp>` |

---

## Architecture

MAKER implements a three-phase orchestration flow:

```
┌─────────────────────────────────────────────────────────────┐
│                   CLI (cli.py)                              │
│         - Role-based model selection                        │
│         - Calibration flag                                  │
│         - Domain adapter selection                          │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│            Calibrator (calibration.py)                      │
│  [Optional] Estimate p, calculate optimal K                 │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│         MakerOrchestrator (orchestrator.py)                 │
│              (Non-LLM, Deterministic)                       │
├─────────────────────────────────────────────────────────────┤
│ Phase 1: Direct Solve (low-cost attempt)                    │
│ Phase 2: Recursive Decomposition                            │
│ Phase 3: Finalization (compose → check → verify → gap-fill) │
│ • Progress Tracking (stagnation detection)                  │
│ • Cycle Detection (memoization)                             │
│ • Timeout Handling                                          │
└─────┬──────────┬──────────────┬──────────┬──────────────────┘
      │          │              │          │
      ▼          ▼              ▼          ▼
  Decomposer  Solver      Composer      Completeness
   (LLM)      (LLM)        (LLM)         Checker (LLM)
      │          │              │          │
      │    ┌─────┴──────────┬───┴────┐     │
      │    │                │        │     │
      ▼    ▼                ▼        ▼     ▼
    DecompositionDiscriminator    SolutionDiscriminator
         (ahead-by-K)                 (ahead-by-K)
              │                            │
              └────┬─────────────┬─────────┘
                   │             │
                   ▼             ▼
              RedFlagGuard    StateVerifier
               (pre-vote)      (post-vote)
                   │             │
                   └─────┬───────┘
                         │
                         ▼
                  GlobalVerifier
               (final sanity checks)
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    DomainAdapter                 EventLogger
     (pluggable)                    (JSONL)
```

---

## Agent Types

### 0. Calibrator (`calibration.py`)
**Code + LLM sampling, optional pre-run phase**

Purpose: Estimate per-step success probability (p) and calculate optimal ahead-by-K value.

Configuration (`CalibrationConfig`):
- `enabled`: Whether to run calibration phase
- `samples`: Number of calibration samples (default: 5)
- `target_error_rate`: Target error rate (default: 0.01)
- `fallback_p`: Assumed success probability if skipped (default: 0.7)
- `max_k`: Maximum ahead-by-k value (default: 5)

Key Methods:
- `calculate_optimal_k(p, target_error_rate)`: Uses formula `error_rate ≈ ((1-p)/p)^k`
- `estimate_p_from_samples()`: Bayesian estimation with prior
- `get_default_result()`: Returns fallback when calibration skipped

### 1. Orchestrator (`orchestrator.py`)
**Non-LLM, deterministic, auditable**

Responsibilities:
- Owns global task state
- Dispatches agents in three phases (Direct → Decompose → Finalize)
- Controls retries, limits, escalation
- Persists full execution trace
- Detects cycles and stagnation
- Routes problems to decomposition or atomic solving

The Orchestrator is the ONLY component that mutates global state. It never uses an LLM.

### 2. Decomposition Agents (`decomposer.py`)
**LLM-based**

Input: A problem description
Output:
```json
{
  "subproblem_A": "...",
  "subproblem_B": "...",
  "compose_fn": "how to combine results",
  "is_atomic": false
}
```

Rules:
- Multiple decompositions are sampled independently
- Each decomposition is voted on before selection
- If `is_atomic: true`, problem goes directly to solvers

### 3. Decomposition Discriminators (`decomposer.py`)
**LLM-based**

Role: Vote among candidate decompositions using ahead-by-K
Output: Selected decomposition structure

### 4. Atomic Solvers (`solver.py`)
**LLM-based**

Role: Solve the smallest indivisible tasks

Constraints:
- Extremely narrow context (only the atomic problem + minimal state)
- Strong output schema (solution + confidence)
- Low token budget
- No awareness of broader task

### 5. Solution Discriminators (`discriminator.py`)
**Deterministic (ahead-by-K voting)**

Role:
- Cluster equivalent solutions (semantic equivalence)
- Vote to determine dominant answer using ahead-by-K

Implementation:
- Uses solution signatures for equivalence (canonical hash)
- Deterministic voting—no LLM involved
- Counts votes per cluster, winner needs K ahead of runner-up

### 6. Red-Flag Guard (`red_flag.py`)
**Code + Optional LLM**

Purpose: Detect correlated failure signals before voting

Hard rejection triggers:
- Invalid schema
- Missing required fields
- Excessive length
- Repetition / looping patterns
- Self-contradiction
- Tool leakage / meta chatter
- Confidence below threshold

**Rejected outputs are discarded silently** - no repair, no feedback to the model.

### 7. Verifier (`verify.py`)
**Code only, NEVER LLM**

Two classes:
- **StateVerifier**: Step-level validation
  - Validate solution correctness (non-empty, work shown)
  - Detect duplicate solutions via signature tracking
  - Score decompositions and solutions for selection
  - Validate decomposition quality (no tautologies)

- **GlobalVerifier**: Basic sanity checks + sympy validation
  - Verify answer not empty
  - Symbolic math validation using sympy (if points provided)

### 8. Completeness Checker (`completeness.py`)
**LLM-based**

Role: Verify that the answer addresses ALL task requirements

Input: Task + Answer + State
Output:
```json
{
  "requirements": [
    {"requirement": "find intersection points", "status": "ADDRESSED", "reason": "..."},
    {"requirement": "show sum of vectors", "status": "MISSING", "reason": "..."}
  ],
  "complete": false,
  "missing_work": ["Calculate the vector sum of the intersection points"]
}
```

If incomplete, the orchestrator spawns atomic solvers for each `missing_work` item, then recomposes.

### 10. Final Composer (`composer.py`)
**LLM-based**

Role: Compose solved subproblems into a coherent final answer

Input: Task + TaskState with solved subproblems
Output:
```json
{
  "answer": "direct response to task",
  "confidence": 0.95,
  "support": {
    "summary": "...",
    "equations": ["y = 2x + 1", ...],
    "points": [{"label": "P1", "values": {"x": 1, "y": 3}}]
  }
}
```

### 11. Progress Tracker (integrated in `orchestrator.py`)
**Code only, NEVER LLM**

Responsibilities:
- Track state changes via content hashing
- Detect stagnation (no progress after N rounds)
- Trigger escalation when stuck (force finalization)

Implementation: `ProgressTracker` dataclass in orchestrator.py with:
- `record(state)`: Hashes semantic content (facts, solutions, draft)
- `stagnant()`: Returns true after `max_stagnant_rounds` without change

### 12. Domain Adapters (`adapters/`)
**Pluggable verification and composition system**

Purpose: Domain-specific verification, evidence extraction, and result composition.

Base Class (`adapters/base.py`):
```python
class BaseDomainAdapter(ABC):
    name: str
    def verify_solution(self, solution: str, context: Dict) -> Tuple[bool, str]
    def extract_evidence(self, text: str) -> Dict[str, Any]
    def compose_results(self, results: List[str], compose_fn: str) -> str
    def get_red_flag_patterns(self) -> List[RedFlagPattern]
    def get_calibration_problems(self) -> List[str]
```

Built-in Adapters:
- **DefaultAdapter** (`adapters/default.py`): Basic text verification, concatenation-based composition
- **MathAdapter** (`adapters/math.py`): Math-specific with sympy integration, equation/point extraction

Usage:
```bash
python -m swarmmaker "task" --domain math
python -m swarmmaker "task" --domain default
```

---

## The MAKER Step Loop

For each atomic step:

1. **Sample**: Generate N independent candidate outputs from solvers
2. **Filter**: Discard red-flagged candidates (silent rejection)
3. **Vote**: Count equivalent outputs, continue until `winner_count >= max(other_counts) + K`
4. **Commit**: Apply the winning solution
5. **Verify**: Reject and retry if state validation fails

---

## Orchestration Phases

### Direct Solve Phase
Before decomposition, the orchestrator attempts a low-cost direct solve:
- Uses smaller batch size and fewer rounds
- Lower temperature for consistency
- If verified successfully, skips decomposition entirely
- Efficient for simple tasks that don't need breaking down

### Decomposition Phase
If direct solve fails or cannot be verified:
- Recursive decomposition until atomic problems
- Each subproblem solved independently
- Results composed via `compose_fn`

### Finalization Phase
After solving (direct or decomposed), a 2-attempt loop:
1. FinalComposer creates coherent answer from state (LLM)
2. CompletenessChecker verifies all requirements addressed (LLM)
3. If complete → GlobalVerifier validates (code)
4. If incomplete → Atomic solve for each `missing_work` item → Recompose

Fallbacks:
- If LLM composer fails, code-based composition from `draft_answer`
- Emergency finalization returns draft with low confidence (0.3)

### Stagnation Detection
ProgressTracker monitors for stuck states:
- Hashes state content after each step
- If hash unchanged for `max_stagnant_rounds`, triggers escalation
- Escalation: attempts one more atomic solve with adjusted temperature, then forces finalization

---

## Voting: First-to-Ahead-by-K

```
winner_count >= max(other_counts) + K
```

**Parameters**:
- `K`: confidence margin (default: 2)
- `max_rounds`: safety bound on voting rounds
- `batch_size`: parallel sampling size

**Escalation options** (when consensus not reached):
- Increase diversity (temperature)
- Re-prompt with different phrasing
- Switch to stronger model
- Force further decomposition

---

## Decomposition Strategy

```python
def solve(problem):
    if is_atomic(problem):
        return solve_atomic(problem)  # sampling + voting
    else:
        decomposition = select_decomposition(problem)  # sampling + voting
        result_a = solve(decomposition.subproblem_A)
        result_b = solve(decomposition.subproblem_B)
        return decomposition.compose_fn(result_a, result_b)
```

**Key rule**: Only atomic problems are solved directly by LLMs.

---

## File Structure

```
src/swarmmaker/
├── __main__.py           # Entry point
├── cli.py                # CLI with Typer
│
├── orchestrator.py       # Non-LLM coordinator (state machine) + ProgressTracker
├── schemas.py            # Pydantic models, configs, contracts
├── calibration.py        # Pre-run calibration for optimal K
│
├── decomposer.py         # Decomposition agents
├── solver.py             # Atomic solver agents
├── discriminator.py      # Solution equivalence discriminators (deterministic)
├── composer.py           # Final answer composer (LLM-based)
├── completeness.py       # Completeness checker (LLM-based)
│
├── red_flag.py           # Red-flag filtering (pre-vote rejection)
├── verify.py             # Code-only sanity checks + sympy validation
├── voting.py             # Ahead-by-K consensus engine
│
├── llm.py                # OpenRouter LLM client + metrics
├── io.py                 # Event logging (JSONL) + results
│
├── adapters/             # Domain-specific verification
│   ├── __init__.py       # Plugin registry and factory
│   ├── base.py           # BaseDomainAdapter abstract class
│   ├── default.py        # Basic text verification
│   └── math.py           # Math-specific with sympy
│
└── (optional tracing via LangSmith)
```

---

## Schemas

### Configuration Models

```python
class ModelRoles(BaseModel):
    reasoning: str   # For decomposition, composition, completeness (default: claude-sonnet-4)
    execution: str   # For atomic solving (default: claude-sonnet-4)

class ThresholdConfig(BaseModel):
    # Voting thresholds
    min_samples_for_confidence: int = 3
    temperature_first_vote: float = 0.0
    temperature_subsequent: float = 0.1
    # Red-flag limits
    max_solution_tokens: int = 750
    max_solution_chars: int = 3000
    min_confidence: float = 0.3
    # State management
    max_facts: int = 5
    max_notes: int = 20
    max_work_shown_chars: int = 1200

class CalibrationConfig(BaseModel):
    enabled: bool = False
    samples: int = 5
    target_error_rate: float = 0.01
    fallback_p: float = 0.7
    max_k: int = 5

class SwarmConfig(BaseModel):
    models: ModelRoles
    thresholds: ThresholdConfig
    calibration: CalibrationConfig
    batch_size: int = 4
    ahead_by: int = 2
    max_rounds: int = 5
    max_depth: int = 6
    # ... additional runtime parameters
```

### Domain Models

```python
class DecompositionProposal(BaseModel):
    subproblem_a: str
    subproblem_b: str
    compose_fn: str           # How to combine results
    is_atomic: bool           # True if problem cannot be further decomposed
    rationale: str            # Why this decomposition
    # Property: signature (canonical JSON hash)

class AtomicSolution(BaseModel):
    solution: str             # The actual answer
    confidence: float         # 0.0 to 1.0
    work_shown: str           # Intermediate steps (for auditability)
    # Property: signature (canonical hash for voting)

class ProblemFact(BaseModel):
    problem: str
    solution: str
    work_shown: str
    confidence: float
    depth: int

class FinalSupport(BaseModel):
    summary: str
    equations: List[str]
    points: List[SupportPoint]

class FinalAnswer(BaseModel):
    answer: str               # Direct response to the original task
    confidence: float         # 0.0 to 1.0
    support: Optional[FinalSupport]  # Structured evidence

class CompletenessResult(BaseModel):
    requirements: List[RequirementStatus]  # Status of each task requirement
    complete: bool            # True if all requirements addressed
    missing_work: List[str]   # Specific tasks to fill gaps
```

### TaskState
```python
class TaskState(BaseModel):
    task: str                 # Original problem
    decomposition_tree: dict  # Recursive structure
    solved_subproblems: dict  # problem_id -> solution
    facts: dict               # step_id -> ProblemFact (capped to max_facts)
    current_problem: str      # Active subproblem
    depth: int                # Recursion depth
    notes: list               # Progress notes and warnings
    draft_answer: str         # Best current answer candidate
    progress_version: int     # Incremented on state changes
    progress_hash: str        # Content hash for stagnation detection
```

### Execution Trace Models
```python
class StepTrace(BaseModel):
    step_id: str
    kind: str                 # "decomposition" or "atomic"
    problem: str
    depth: int
    candidates: List
    chosen: Any
    votes: Dict
    notes: List[str]
    rejections: List

class RunStats(BaseModel):
    elapsed_s: float
    llm_calls: int
    tokens_in: int
    tokens_out: int
    retries: int
    consensus_votes: int
    aborted_reason: Optional[str]

class RunResult(BaseModel):
    task: str
    final_answer: str
    final_payload: FinalAnswer
    steps: List[StepTrace]
    stats: RunStats
    artifacts: Dict
    created_at: str
```

---

## Red-Flag Patterns

The red-flag guard rejects outputs matching these patterns:

| Pattern | Description |
|---------|-------------|
| `schema_invalid` | JSON doesn't match expected schema |
| `missing_field` | Required field is empty/null |
| `excessive_length` | Output > max_tokens threshold |
| `repetition` | Same phrase repeated >3 times |
| `meta_chatter` | Contains "I think", "Let me", "As an AI" |
| `low_confidence` | Confidence < 0.3 |
| `self_contradiction` | Contradicts own statements |
| `placeholder` | Contains "TODO", "...", "<your answer>" |

---

## Observability

### Per-Step Logging
- Prompt hash
- All raw outputs (pre-filtering)
- Red-flag reasons for rejected outputs
- Vote counts over time
- Token lengths
- Rounds to convergence

### Global Metrics
- Avg samples per step
- Red-flag rejection rate
- Max rounds per step
- Failure recoveries
- Cost per solved task

### Output Files
Each run creates `./runs/<timestamp>/`:
- `events.jsonl`: Append-only event log
- `result.json`: Final structured result
- `result.md`: Human-readable summary

---

## Key Design Principles

### Independence
- Each candidate solution is sampled independently
- No shared scratchpads or stateful reasoning between samples
- Workers have no memory of other workers' outputs

### Strict Rejection
- Invalid outputs are **discarded**, not fixed
- Repairs introduce correlated failures
- Red-flagged outputs never reach voting

### Statistical Correctness
- System works if: `P(correct) > P(any specific wrong answer)`
- Errors are acceptable if not perfectly correlated
- Voting continues until confidence, not exhaustion

### Deterministic Orchestration
- Orchestrator and Verifier are pure code (no LLM)
- All state transitions are auditable
- Full execution trace is reproducible

---

## What MAKER Is NOT

- NOT chain-of-thought (no long reasoning chains)
- NOT tool-heavy planning (tools are atomic, not plans)
- NOT "one smart agent" (many dumb agents, voted)
- NOT self-reflection loops (rejection, not repair)

**Structure replaces intelligence.**

---

## When to Use MAKER

**Use MAKER if**:
- Tasks exceed dozens of steps
- Single-agent reasoning degrades at scale
- Errors are catastrophic (need near-zero error rate)
- Verifiability exists at step level

**Do NOT use MAKER if**:
- Task is short (< 5 steps)
- Steps are not independently verifiable
- Decomposition is impossible
- Speed matters more than correctness

---

## Common Issues

### No Consensus Reached
**Symptom**: Max rounds exceeded without ahead-by-K winner

**Solutions**:
- Increase batch_size to get more samples
- Raise temperature for diversity
- Check if problem is too ambiguous for atomic solving
- Force further decomposition

### Red-Flag Rejection Rate Too High
**Symptom**: Most outputs discarded before voting

**Solutions**:
- Check rejection reasons in events.jsonl
- Simplify atomic problems (decompose further)
- Use stronger solver model
- Relax non-critical red-flag rules

### Decomposition Loops
**Symptom**: Same problem keeps decomposing without becoming atomic

**Solutions**:
- Reduce `--max-depth` (default: 6)
- Improve atomicity detection in decomposer prompt
- Cycle detection automatically forces atomic solving

### Stagnation / No Progress
**Symptom**: State hash unchanged for multiple rounds

**Solutions**:
- ProgressTracker will auto-escalate and force finalization
- Check if problem requires domain-specific adapter
- Enable `--calibrate` to tune ahead-by-K

### Verifier Rejects All Solutions
**Symptom**: Step retries exhaust without valid solution

**Solutions**:
- Check verifier rules aren't too strict
- Verify problem is actually solvable
- Check if domain constraints are correctly defined
