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
┌──────────────┐
│  Orchestrator│  (Non-LLM, deterministic)
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│   Problem Decomposition │◄──────────────┐
└──────────┬──────────────┘               │
           ▼                              │
┌─────────────────────────┐              │
│   Atomic Step Execution │              │
└──────────┬──────────────┘              │
           ▼                              │
┌─────────────────────────┐              │
│  Red-Flag Filtering     │              │
└──────────┬──────────────┘              │
           ▼                              │
┌─────────────────────────┐              │
│  Voting (Ahead-by-K)    │──────────────┘
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ State Update + Verify   │  (Non-LLM, deterministic)
└─────────────────────────┘
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

### 7. Verifier (`verifier.py`)
**Code only, NEVER LLM**

Responsibilities:
- Validate step correctness against known invariants
- Check state transitions are legal
- Enforce domain constraints (balanced equations, valid syntax, etc.)
- Reject and trigger retry if validation fails

---

## The MAKER Step Loop

For each atomic step:

1. **Sample**: Generate N independent candidate outputs from solvers
2. **Filter**: Discard red-flagged candidates (silent rejection)
3. **Vote**: Count equivalent outputs, continue until `winner_count >= max(other_counts) + K`
4. **Commit**: Apply the winning solution
5. **Verify**: Reject and retry if state validation fails

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
├── state.py              # Global task state (immutable snapshots)
│
├── decomposer.py         # Decomposition agents + discriminators
├── solver.py             # Atomic solver agents
├── discriminator.py      # Solution equivalence + voting
│
├── red_flag.py           # Red-flag filtering (pre-vote rejection)
├── verifier.py           # Code-only state validation
├── voting.py             # Ahead-by-K consensus engine
│
├── llm.py                # OpenRouter LLM client
├── schemas.py            # Pydantic models (contracts)
│
├── callbacks.py          # Rich terminal UI
├── io.py                 # Event logging (JSONL) + results
└── langsmith.py          # Tracing integration
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

### TaskState
```python
class TaskState(BaseModel):
    task: str                 # Original problem
    decomposition_tree: dict  # Recursive structure
    solved_subproblems: dict  # problem_id -> solution
    current_problem: str      # Active subproblem
    depth: int                # Recursion depth
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
