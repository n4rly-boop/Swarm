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

# Custom configuration
python -m swarmmaker "task" \
  --model-decomposer anthropic/claude-3-haiku \
  --model-solver qwen/qwen2.5-coder-7b-instruct \
  --batch-size 5 \
  --ahead-by 2 \
  --max-rounds 10
```

### Environment Variables
```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional
export LANGSMITH_API_KEY=...                              # optional tracing
export LANGCHAIN_PROJECT="maker"
```

---

## Architecture

MAKER implements the architecture defined in `MAKER.md`:

```
┌──────────────────────────────────────────────────────────────┐
│                    Orchestrator (Non-LLM)                    │
│  - Owns global TaskState                                     │
│  - Dispatches agents, controls flow                          │
│  - Tracks progress, detects stagnation                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌────────────────┐
│  Direct Solve   │ │ Decomposer  │ │ FinalComposer  │
│  (fast path)    │ │   (LLM)     │ │    (LLM)       │
└────────┬────────┘ └──────┬──────┘ └───────┬────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│               Atomic Solver (LLM, batch sampling)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌────────────────┐
│  Red-Flag Guard │ │ StateVerify │ │ Discriminator  │
│    (Code)       │ │   (Code)    │ │ (Ahead-by-K)   │
└─────────────────┘ └─────────────┘ └────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   GlobalVerifier       │
              │ (Code + sympy math)    │
              └────────────────────────┘
```

---

## Agent Types

### 1. Orchestrator (`orchestrator.py`)
**Non-LLM, deterministic, auditable**

Responsibilities:
- Owns global task state
- Dispatches agents in correct sequence
- Controls retries, limits, escalation
- Persists full execution trace
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
**LLM or Code**

Role:
- Cluster equivalent solutions (semantic equivalence)
- Vote to determine dominant answer

May be replaced by:
- Exact match
- Semantic equivalence classifier
- Domain-specific verifier (math, code)

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

### 11. Progress Tracker (`progress.py`)
**Code only, NEVER LLM**

Responsibilities:
- Track state changes via content hashing
- Detect stagnation (no progress after N rounds)
- Trigger escalation when stuck (force finalization)

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
After solving (direct or decomposed):
- FinalComposer creates coherent answer from state
- GlobalVerifier validates against original task
- Retry with feedback if verification fails (max 2 attempts)

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
├── config.py             # Environment/settings
│
├── orchestrator.py       # Non-LLM coordinator (state machine)
├── schemas.py            # Pydantic models (contracts) + TaskState
│
├── decomposer.py         # Decomposition agents
├── solver.py             # Atomic solver agents
├── discriminator.py      # Solution equivalence discriminators
├── composer.py           # Final answer composer (LLM-based)
├── completeness.py       # Completeness checker (LLM-based)
│
├── red_flag.py           # Red-flag filtering (pre-vote rejection)
├── verify.py             # Code-only sanity checks + sympy validation
├── voting.py             # Ahead-by-K consensus engine
├── progress.py           # Progress tracking + stagnation detection
│
├── llm.py                # OpenRouter LLM client
├── io.py                 # Event logging (JSONL) + results
└── (optional tracing via LangSmith)
```

---

## Schemas

### DecompositionProposal
```python
class DecompositionProposal(BaseModel):
    subproblem_a: str
    subproblem_b: str
    compose_fn: str           # How to combine results
    is_atomic: bool           # True if problem cannot be further decomposed
    rationale: str            # Why this decomposition
```

### AtomicSolution
```python
class AtomicSolution(BaseModel):
    solution: str             # The actual answer
    confidence: float         # 0.0 to 1.0
    work_shown: str           # Intermediate steps (for auditability)
```

### FinalAnswer
```python
class FinalAnswer(BaseModel):
    answer: str               # Direct response to the original task
    confidence: float         # 0.0 to 1.0
    support: Optional[FinalSupport]  # Structured evidence (equations, points)
```

### CompletenessResult
```python
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
- Add max_depth limit to orchestrator
- Improve atomicity detection in decomposer prompt
- Force atomic solving after N decomposition attempts

### Verifier Rejects All Solutions
**Symptom**: Step retries exhaust without valid solution

**Solutions**:
- Check verifier rules aren't too strict
- Verify problem is actually solvable
- Check if domain constraints are correctly defined
