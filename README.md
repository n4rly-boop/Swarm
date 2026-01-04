# SwarmMaker

SwarmMaker now follows the MAKER architecture from `MAKER.md`: maximal decomposition, ahead-by-K voting, strict red-flag filtering, and a deterministic orchestrator.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m swarmmaker --help
```

Set the OpenRouter key before running:

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional override
```

Use `--dry-run` to exercise the loop without network calls:

```bash
python -m swarmmaker "draft a launch plan" --dry-run
```

## CLI

The CLI exposes MAKER controls directly:

- `--model-decomposer`, `--model-solver`, `--model-discriminator` select the LLMs for decomposition, atomic solving, and tiebreaking.
- `--batch-size`, `--max-rounds`, and `--ahead-by` control sampling, filtering, and voting.
- `--max-depth` bounds recursion, while `--max-total-tokens` + `--timeout-seconds` bound budget.
- `--structured-mode` chooses between provider-side JSON schema or json_object formats.
- `--log-dir` overrides the default `./runs/<timestamp>` artifact folder.

## Outputs

Each run writes:

- `events.jsonl` – append-only event log of decompositions, votes, rejections, and solutions.
- `result.json` – structured `RunResult` (task, stats, per-step traces, artifact locations).
- `result.md` – concise Markdown summary of the task, final answer, stats, and paths.

The orchestrator itself is deterministic and non-LLM: only the decomposer/solver/discriminator agents use LLM calls, while red-flag filtering and verification stay code-only.
