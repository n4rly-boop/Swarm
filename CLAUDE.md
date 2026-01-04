# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running the Application
```bash
# Show available options
python -m swarmmaker --help

# Run with a task
python -m swarmmaker "your task here"

# Run in dry mode (no API calls, mock data only)
python -m swarmmaker "draft a launch plan" --dry-run

# Run with custom configuration
python -m swarmmaker "your task" \
  --model-planner google/gemini-2.5-flash-preview-09-2025 \
  --model-worker qwen/qwen2.5-coder-7b-instruct \
  --swarm-size 5 \
  --max-steps 12 \
  --ahead-by 2 \
  --show-rationale
```

### Environment Variables
Required for non-dry-run execution:
```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional override
export LANGSMITH_API_KEY=...                            # optional tracing
export LANGCHAIN_PROJECT="swarmmaker-mvp"
```

## Architecture

### Multi-Agent Flow
SwarmMaker implements a **planner-voter-aggregator-verifier** architecture using LangGraph:

1. **Planner** (`plan_node`) - Coordinates the overall strategy by analyzing task progress and determining the next **granular, atomic** goal for worker agents. Creates single-purpose steps that small models can execute (e.g., "Simplify x²-5x+7=2x+1 to standard form" rather than "solve the system"). Outputs a `PlannerStep` with the goal, stop condition, and token budget.

2. **Voters/Workers** (`propose_node`) - Multiple worker agents execute in parallel (default 5), each proposing an `Action` to achieve the planner's goal. Workers use **context-dependent prompts**:
   - When `stop_condition="done"`: Enforces FINAL action with complete answer, rejects all other action types
   - When `stop_condition="continue"`: Accepts NOTE/DO actions with emphasis on showing actual work (equations, calculations) not descriptions
   - Individual worker actions are logged to `events.jsonl` with voter IDs for traceability
   - Uses async execution with early consensus termination when a candidate is ahead by K votes (configurable via `--ahead-by`)

3. **Consensus Engine** (`aggregate_node`) - Aggregates worker votes using action signatures (canonical JSON of action_type+args). If no clear winner emerges, escalates to the judge agent.

4. **Judge** (`_call_judge`) - Breaks ties between top-2 candidates when consensus is not reached. Returns the selected action signature.

5. **Verifier** (`verify_apply_node`) - Validates the chosen action against structural requirements, planner constraints, and **content quality**:
   - Structural: step_id match, action_type validity, args format
   - Constraint: FINAL required when stop_condition="done"
   - Content quality: Rejects vague intentions ("I will...", "should..."), rejects actions that restate goal without actual work
   - On success, applies state mutations (notes, draft_answer, final_answer)
   - On failure, triggers retry logic (max 2 retries per step)

### Graph Flow
```
check → plan → propose → aggregate → verify_apply → check (loop)
                                                   ↓
                                                 final
```

The `check_node` enforces budgets (max_steps, max_total_tokens, max_wall_seconds) and aborts if limits are reached.

### Key Components

**graph.py** - LangGraph workflow definition. Core state machine with nodes for each agent type. The `RuntimeContext` dataclass bundles all shared services (LLM client, consensus engine, verifier, display, event logger, metrics, LangSmith manager).

**schemas.py** - Pydantic models defining the contract between agents:
- `PlannerStep` - Planner output specifying next worker goal (designed for granular, atomic steps)
- `Action` - Worker proposal with action_type:
  - `FINAL` - Complete answer to the task (required when stop_condition="done")
  - `NOTE` - Observation or intermediate finding with actual results
  - `DO` - Performed action/computation with actual work shown
  - `ASK_CLARIFY` - Discouraged; only used when information is truly missing (validated strictly)
- `Action.signature` property generates canonical JSON for deduplication
- `StepRecord` - History entry for each completed step
- `RunResult` - Final output with task, answer, steps, stats, artifacts

**llm.py** - Unified LLM client using OpenRouter. Handles structured output via `response_format` with two modes:
- `json_schema` (default) - Strict schema enforcement via provider
- `json_object` - Fallback for providers without schema support
- Includes retry logic, token tracking, and streaming support

**consensus.py** - Implements ahead-by-K voting with early termination. Workers can be stopped mid-flight once a candidate reaches K votes ahead of the runner-up.

**verify.py** - `ActionVerifier` enforces structural, constraint, and content validation:
- **Structural**: Step ID matching between planner and worker, action_type validity, args format
- **Constraint**: FINAL action required when planner sets stop_condition="done"
- **Content Quality** (new):
  - Rejects vague intentions: actions starting with "I will", "we should", "going to", etc.
  - Rejects goal restatement: actions that repeat planner's goal without showing actual work
  - Requires evidence of work: must contain equations, calculations, or specific values (=+-*/^() or digits)
- **Loop Prevention**: Duplicate signature detection to prevent repeated identical actions
- **Action-specific validation**: FINAL requires non-empty content, DO requires non-empty args, ASK_CLARIFY discouraged

**callbacks.py** - Rich UI with live updating panels. Uses thread-safe event queue to coordinate between async LLM calls and the main render loop. Displays planner, voters, aggregate, judge, verifier, and final panels plus metrics and event log.

**io.py** - Event logging (JSONL), result serialization (JSON + Markdown). Each run creates a timestamped directory in `./runs/` with `events.jsonl`, `result.json`, and `result.md`.

**cli.py** - Typer-based CLI entrypoint with extensive configuration flags for models, temperatures, budget limits, and LangSmith integration.

**config.py** - Singleton settings loader that reads `.env` file from project root.

### Structured Output System
The LLM client uses provider-level structured output enforcement via `response_format`:
- Schemas are passed to the provider, not in the prompt
- Two modes: `json_schema` (strict) and `json_object` (lenient)
- All agent outputs (PlannerStep, Action, JudgeSelection) are validated with Pydantic
- Failed parses raise `SchemaValidationError` with agent context

### Retry and Error Handling
- Worker errors during propose phase are logged but don't fail the step
- Verifier rejections trigger `retry_step=True`, which re-runs the same planner step
- **Retry counting**: Incremented only in `verify_apply_node` to avoid double-counting (bug fix)
- Max retries (default 2) enforced per-step: allows 2 total attempts (original + 1 retry)
- Transient LLM errors (timeout, rate limit) trigger exponential backoff in `llm.py`
- Content validation failures (vague actions, goal restatement) trigger retries just like structural failures

### LangSmith Integration
When `LANGSMITH_API_KEY` is set, creates a parent "SwarmMaker" run that traces all LLM calls. Each agent call is tagged with agent name, stage, step_id, and voter_id for filtering in the LangSmith UI.

### Worker Prompt System
Workers receive context-dependent prompts based on the planner's stop_condition:

**When FINAL Required** (`stop_condition="done"`):
- Shows only FINAL action schema with exact step_id pre-filled
- Emphasizes that ANY other action type will be REJECTED
- Instructs to include all work, calculations, and reasoning in args.content
- Explicitly warns against using placeholder text

**When FINAL Not Required** (`stop_condition="continue"`):
- Shows NOTE and DO action types (ASK_CLARIFY omitted from schema)
- Provides GOOD examples showing actual work: `"x^2 - 7x + 6 = (x-1)(x-6)"`
- Provides BAD examples to avoid: `"I will factor the equation"`
- Emphasizes showing equations/calculations, not descriptions
- Rejects vague intentions through verifier

### Design for Small Models
The system is designed to work with smaller, more cost-effective worker models (7B-13B parameters):

**Granular Steps**: Planner creates atomic tasks that require minimal computation (e.g., "Factor x²-7x+6" not "Factor and solve and find y")

**Clear Examples**: Worker prompts include concrete examples of what actual work looks like vs vague descriptions

**Progressive Validation**: Verifier rejects non-progressive content, forcing models to retry with better output

**Structured Output**: Provider-level schema enforcement ensures valid JSON even from smaller models

**Early Termination**: Ahead-by-K voting stops worker execution as soon as consensus emerges, reducing costs

### Dry Run Mode
Setting `--dry-run` bypasses all LLM calls and generates deterministic mock data based on agent type. Useful for testing UI, event logging, and flow logic without API costs.

## Output Structure
Each run produces a timestamped directory (e.g., `./runs/20250104-123456/`) containing:
- `events.jsonl` - Append-only log of all events with timestamps, including:
  - `task` - Initial task received
  - `planner_step` - Planner's goal and configuration for each step
  - `worker_action` - Individual worker action proposals (with voter ID: voter_1, voter_2, etc.)
  - `worker_invalid_final` - Worker produced non-FINAL when FINAL required
  - `voter_batch` - Aggregated worker vote counts
  - `consensus` - Consensus engine results
  - `judge_choice` - Judge's selection when needed
  - `verify_reject` - Verifier rejected action (with reason)
  - `action_applied` - Action passed verification and was applied
  - `plan_retry` - Step retry triggered
  - `abort` - Run aborted (with reason)
  - `final` - Run completed
- `result.json` - Structured RunResult payload with full step history
- `result.md` - Human-readable summary with task, answer, stats, and artifact links

## Common Issues and Debugging

### Workers Producing Vague Actions
**Symptom**: Workers say "I will solve..." or "We should factor..." instead of showing actual work

**Causes**:
- Worker model too weak for the task complexity
- Planner creating compound goals (e.g., "factor AND solve")

**Solutions**:
- Check `events.jsonl` for `verify_reject` events with reason "action too vague"
- Increase `--max-steps` to allow more granular breakdown
- Try stronger worker model: `--model-worker google/gemini-2.5-flash-preview-09-2025`
- Check planner goals are atomic (each should do ONE thing)

### Retries Exhausted Without Progress
**Symptom**: Run aborts with "max retries reached" after minimal attempts

**Causes**:
- Workers not understanding the task
- Model not following structured output format
- Content validation rejecting all attempts

**Solutions**:
- Check `events.jsonl` for `verify_reject` reasons
- Look for `worker_action` events to see what workers actually produced
- If workers produce wrong step_id: model may not support strict schema mode, try `--structured-mode json_object`
- If all actions identical: increase temperature `--temperature-worker 0.9`

### Workers Repeating Same Content
**Symptom**: Multiple steps produce identical or very similar actions

**Causes**:
- Duplicate signature detection not triggering (signatures differ slightly)
- Workers not using context from previous steps
- Planner not incorporating notes into next goal

**Solutions**:
- Check notes_snapshot in `result.json` steps - are notes accumulating?
- Verify planner is seeing context: recent notes appear in planner prompt
- Check if worker confidence is very low - may indicate confusion

### Final Answer is Placeholder Text
**Symptom**: Final answer is literally `"<your complete final answer here>"` or similar

**Cause**: Worker copied example text from prompt instead of generating real answer

**Solution**: This should be fixed in current version - update to latest code

### Tracing Individual Workers
Use `events.jsonl` to trace individual worker behavior:
```bash
# See all actions from voter_1
jq 'select(.agent == "voter_1")' runs/TIMESTAMP/events.jsonl

# See all verify_reject events with reasons
jq 'select(.type == "verify_reject") | {step_id, reason: .payload.reason}' runs/TIMESTAMP/events.jsonl

# Count actions by type
jq -r '.payload.action.action_type // empty' runs/TIMESTAMP/events.jsonl | sort | uniq -c
```
