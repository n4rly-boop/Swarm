# SwarmMaker

MVP multi-agent CLI orchestrator that runs a planner-voter-aggregator-verifier flow with Rich streaming UI and LangSmith tracing.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m swarmmaker --help
```

Set required environment variables before running:

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional override
export LANGSMITH_API_KEY=...                            # optional tracing
export LANGCHAIN_PROJECT="swarmmaker-mvp"
```

Run in dry mode to see the UI without hitting the API:

```bash
python -m swarmmaker "draft a launch plan" --dry-run
```

## Running

- Use `python -m swarmmaker "your task here"` or the `swarmmaker` console script.
- Tune models and limits with flags such as `--model-planner`, `--swarm-size`, `--ahead-by`, `--max-steps`, `--max-total-tokens`, `--max-wall-seconds`, `--max-retries`, and temperature options per agent.
- `--log-dir` lets you override the default `./runs/<timestamp>` folder for artifacts, while `--project-name` syncs with LangSmith projects.
- Enable `--show-rationale` to include one-sentence voter rationales, and `--dry-run` to skip LLM calls while still exercising Rich streaming and logging.

## Outputs

Each run writes:

- `events.jsonl` – append-only event log.
- `result.json` – structured `RunResult` payload with step history, stats, and artifact paths.
- `result.md` – Markdown summary containing the original task, final answer, execution summary, and stats (including LangSmith URL when available).

## LangSmith

Tracing is enabled automatically when `LANGSMITH_API_KEY` is set. Set `--project-name` or `LANGCHAIN_PROJECT` to control project attribution. Result metadata includes a LangSmith run URL when available.
