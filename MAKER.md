# MAKER: General-Purpose Massively Decomposed Agentic System

## Overview

MAKER (Maximal Agentic decomposition + first-to-Ahead-by-K Error correction + Red-flagging)
is a framework for executing **very long-horizon tasks** with **near-zero error rate**
by converting them into **many tiny, independently correctable steps**.

Core idea:
> Never trust a single long reasoning chain.  
> Trust *many small steps*, each protected by **sampling, filtering, and voting**.

This document describes a **general-purpose MAKER system**, not task-specific (e.g. Hanoi).

---

## Core Principles

1. **Maximal Decomposition**
   - Break tasks until steps are *atomic*
   - Atomic = high probability of correctness per attempt

2. **Independence**
   - Each candidate solution is sampled independently
   - No shared scratchpads or stateful reasoning

3. **Error Correction, Not Error Avoidance**
   - Expect errors
   - Detect and correct them statistically

4. **Strict Rejection Beats Repair**
   - Invalid outputs are discarded, not fixed
   - Repairs introduce correlated failures

5. **Confidence-Based Stopping**
   - Stop only when statistical dominance is reached

---

## High-Level Architecture

```

┌──────────────┐
│  Orchestrator│
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
│ State Update + Verify   │
└─────────────────────────┘

````

---

## Agent Types

### 1. Orchestrator (Non-LLM)

**Responsibilities**
- Owns global task state
- Dispatches agents
- Controls retries, limits, and escalation
- Persists full execution trace

**Must be deterministic and auditable**

---

### 2. Decomposition Agents (LLM)

**Input**
- A problem description

**Output**
```json
{
  "subproblem_A": "...",
  "subproblem_B": "...",
  "compose_fn": "how to combine results",
  "stop": false
}
````

**Notes**

* Multiple decompositions are sampled
* No guarantee any single one is correct

---

### 3. Decomposition Discriminator Agents (LLM)

**Role**

* Vote among candidate decompositions
* Uses ahead-by-K voting

**Output**

* Selected decomposition structure

---

### 4. Atomic Solver Agents (LLM)

**Role**

* Solve the smallest indivisible tasks

**Constraints**

* Extremely narrow context
* Strong output schema
* Low token budget

---

### 5. Solution Discriminator Agents (LLM or Code)

**Role**

* Cluster equivalent solutions
* Vote to determine dominant answer

**May be replaced by**

* Exact match
* Semantic equivalence classifier
* Domain-specific verifier

---

### 6. Red-Flag Guard (Code + Optional LLM)

**Purpose**
Detect correlated failure signals.

**Hard rejection triggers**

* Invalid schema
* Missing required fields
* Excessive length
* Repetition / looping
* Self-contradiction
* Tool leakage / meta chatter

> Rejected outputs are **discarded silently**

---

### 7. Verifier (Code)

**Responsibilities**

* Validate step correctness
* Enforce invariants
* Check state transitions

**Never uses an LLM**

---

## The MAKER Step Loop

For each atomic step `t`:

1. **Sample**

   * Generate independent candidate outputs

2. **Filter**

   * Discard red-flagged candidates

3. **Vote**

   * Count equivalent outputs
   * Continue sampling until:

     ```
     winner_count ≥ max(other_counts) + K
     ```

4. **Commit**

   * Apply the winning action

5. **Verify**

   * Reject and retry if state validation fails

---

## Voting Strategy: First-to-Ahead-by-K

**Why**

* Simple majority fails under correlated errors
* Ahead-by-K provides statistical confidence

**Parameters**

* `K`: confidence margin
* `max_rounds`: safety bound
* `batch_size`: parallel sampling size

**Escalation options**

* Increase diversity
* Re-prompt
* Switch model
* Force decomposition

---

## Decomposition Strategy

### Recursive Decomposition

```
Solve(P):
  if atomic(P):
    return solve_atomic(P)
  else:
    D = select_decomposition(P)
    A = Solve(D.subproblem_A)
    B = Solve(D.subproblem_B)
    return D.compose_fn(A, B)
```

**Key rule**

> Only atomic problems are solved directly by LLMs.

---

## Reliability Guarantees (Design Assumptions)

The system works if:

* Each atomic step has:

  ```
  P(correct) > P(any specific wrong answer)
  ```
* Errors are **not perfectly correlated**
* Red-flagging removes high-risk outputs
* Voting continues until confidence, not exhaustion

---

## Observability & Metrics

Required logging per step:

* Prompt hash
* All raw outputs
* Red-flag reasons
* Vote counts over time
* Token lengths
* Rounds to convergence

Global metrics:

* Avg samples per step
* Red-flag rate
* Max rounds per step
* Failure recoveries
* Cost per solved task

---

## What MAKER Is NOT

* ❌ Not chain-of-thought
* ❌ Not tool-heavy planning
* ❌ Not “one smart agent”
* ❌ Not self-reflection loops

MAKER is **statistical correctness via structure**, not intelligence.

---

## When to Use MAKER

Use MAKER if:

* Tasks exceed hundreds or thousands of steps
* Single-agent reasoning degrades
* Errors are catastrophic
* Verifiability exists at step level

Do NOT use MAKER if:

* Task is short
* Steps are not verifiable
* Decomposition is impossible

---

## Summary

MAKER turns LLMs from:

> fragile reasoners
> into
> reliable stochastic components

Reliability comes from:

* decomposition
* independence
* rejection
* voting
* verification

**Structure replaces intelligence.**
