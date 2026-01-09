"""Command-line entrypoint for the MAKER-aligned SwarmMaker runtime."""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .adapters import get_adapter
from .calibration import Calibrator
from .completeness import CompletenessChecker
from .composer import FinalComposer
from .decomposer import Decomposer
from .io import EventLogger, ensure_log_dir, save_json_result, save_markdown_result
from .llm import LLMClient, MetricsTracker
from .orchestrator import (
    MakerOrchestrator,
    MakerRuntime,
    ProgressTracker,
    RunReporter,
)
from .red_flag import RedFlagGuard
from .schemas import (
    CalibrationConfig,
    ModelRoles,
    RunArtifacts,
    RunResult,
    RunStats,
    StructuredMode,
    SwarmConfig,
    ThresholdConfig,
    canonical_json,
    env,
)
from .solver import AtomicSolver
from .verify import GlobalVerifier, StateVerifier
from .voting import DecompositionDiscriminator, SolutionDiscriminator

app = typer.Typer(add_completion=False, help="SwarmMaker MAKER orchestrator CLI.")

# Default models for the two roles
DEFAULT_REASONING_MODEL = "openai/gpt-4.1-mini"
DEFAULT_EXECUTION_MODEL = "qwen/qwen2.5-coder-7b-instruct"


@app.command()
def main(
    task: str = typer.Argument(..., help="Task description to solve."),
    # Model configuration (role-based)
    model_reasoning: str = typer.Option(
        DEFAULT_REASONING_MODEL,
        "--model-reasoning",
        "-r",
        help="Model for decomposition, composition, and completeness.",
    ),
    model_execution: str = typer.Option(
        DEFAULT_EXECUTION_MODEL,
        "--model-execution",
        "-e",
        help="Model for atomic solving.",
    ),
    # Core algorithm parameters
    batch_size: int = typer.Option(4, "--batch-size", help="Samples per round for voting."),
    ahead_by: int = typer.Option(2, "--ahead-by", help="Votes required beyond runner-up."),
    max_rounds: int = typer.Option(5, "--max-rounds", help="Max sampling rounds per stage."),
    max_depth: int = typer.Option(6, "--max-depth", help="Maximum recursion depth."),
    # Resource limits
    max_total_tokens: int = typer.Option(50_000, "--max-total-tokens", help="Budget for all LLM calls."),
    timeout_seconds: int = typer.Option(60, "--timeout-seconds", help="Timeout per LLM call."),
    timeout_total: int = typer.Option(600, "--timeout-total", help="Total orchestrator timeout."),
    # Calibration
    calibrate: bool = typer.Option(False, "--calibrate", help="Run calibration phase before execution."),
    # Domain adapter
    domain: str = typer.Option("default", "--domain", help="Domain adapter (default, math)."),
    # LLM settings
    structured_mode: StructuredMode = typer.Option(
        StructuredMode.json_schema,
        "--structured-mode",
        case_sensitive=False,
        help="Response format enforcement mode.",
    ),
    # Execution control
    dry_run: bool = typer.Option(False, "--dry-run", help="Return deterministic mock results."),
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="Directory for artifacts (defaults to runs/<timestamp>).",
    ),
) -> None:
    """Execute the MAKER step loop for the provided task."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    target_log_dir = ensure_log_dir(log_dir or Path("runs") / timestamp)
    events_path = target_log_dir / "events.jsonl"
    result_json_path = target_log_dir / "result.json"
    result_md_path = target_log_dir / "result.md"

    # Build configuration with new structure
    config = SwarmConfig(
        models=ModelRoles(
            reasoning=model_reasoning,
            execution=model_execution,
        ),
        thresholds=ThresholdConfig(),
        calibration=CalibrationConfig(enabled=calibrate),
        batch_size=batch_size,
        ahead_by=ahead_by,
        max_rounds=max_rounds,
        max_depth=max_depth,
        max_total_tokens=max_total_tokens,
        timeout_seconds=timeout_seconds,
        timeout_total_seconds=timeout_total,
        dry_run=dry_run,
        log_dir=target_log_dir,
        structured_mode=structured_mode,
        domain=domain,
    )

    # Get API key
    openrouter_key = env("OPENROUTER_API_KEY")
    if not config.dry_run and not openrouter_key:
        typer.echo("OPENROUTER_API_KEY is required unless --dry-run is set.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration: {canonical_json(config.model_dump(mode='python'))}")

    # Initialize components
    events = EventLogger(events_path)
    events.log("task", {"task": task}, message="task received", agent="orchestrator")

    metrics = MetricsTracker(config.max_total_tokens)
    llm_client = LLMClient(
        api_key=openrouter_key,
        base_url=env("OPENROUTER_BASE_URL"),
        timeout=config.timeout_seconds,
        metrics=metrics,
        structured_mode=config.structured_mode,
        dry_run=config.dry_run,
    )

    reporter = RunReporter(emit=typer.echo)
    reporter.start(task, config)

    # Get domain adapter
    adapter = get_adapter(domain)

    # Initialize verifiers with thresholds and adapter
    verifier = StateVerifier(thresholds=config.thresholds)
    global_verifier = GlobalVerifier(thresholds=config.thresholds, adapter=adapter)

    # Initialize red-flag guard with thresholds
    red_flag = RedFlagGuard(thresholds=config.thresholds)

    # Initialize discriminators with min_samples from thresholds
    decomposition_discriminator = DecompositionDiscriminator(
        ahead_by=config.ahead_by,
        min_samples=config.thresholds.min_samples_for_confidence,
    )
    solution_discriminator = SolutionDiscriminator(
        ahead_by=config.ahead_by,
        min_samples=config.thresholds.min_samples_for_confidence,
    )

    # Initialize progress tracker
    progress_tracker = ProgressTracker(max_stagnant_rounds=config.max_stagnant_rounds)

    # Build runtime
    runtime = MakerRuntime(
        config=config,
        decomposer=Decomposer(llm_client, config),
        solver=AtomicSolver(llm_client, config),
        decomposition_discriminator=decomposition_discriminator,
        solution_discriminator=solution_discriminator,
        red_flag=red_flag,
        verifier=verifier,
        global_verifier=global_verifier,
        completeness_checker=CompletenessChecker(llm_client, config),
        composer=FinalComposer(llm_client, config),
        logger=events,
        metrics=metrics,
        progress_tracker=progress_tracker,
        adapter=adapter,
        reporter=reporter,
    )

    # Run calibration if enabled
    if config.calibration.enabled:
        typer.echo("Running calibration phase...")
        calibrator = Calibrator(config.calibration)

        # Use adapter's calibration problems if available
        calibration_problems = adapter.get_calibration_problems()
        if calibration_problems:
            typer.echo(f"Using {len(calibration_problems)} calibration problems from {domain} adapter")

        # For now, just calculate optimal k from default p
        calibration_result = calibrator.get_default_result()
        typer.echo(
            f"Calibration: p={calibration_result.estimated_p:.2f}, "
            f"optimal_k={calibration_result.optimal_k}"
        )

        # Update config with calibrated k
        config.ahead_by = calibration_result.optimal_k

    # Run orchestrator
    orchestrator = MakerOrchestrator(runtime)
    start = time.perf_counter()

    try:
        result = orchestrator.run(task)
    except KeyboardInterrupt:
        typer.echo("Interrupted by user.", err=True)
        raise typer.Exit(1)
    finally:
        elapsed = time.perf_counter() - start
        typer.echo(f"Elapsed: {elapsed:.2f}s")

    # Save results
    result.artifacts.events_path = events_path
    result.artifacts.result_md_path = result_md_path
    result.artifacts.result_json_path = result_json_path

    save_json_result(result, result_json_path)
    save_markdown_result(result, result_md_path)

    typer.echo(f"Result saved to {result_json_path}")
    typer.echo(f"Markdown summary: {result_md_path}")

    if result.final_answer:
        typer.echo("\n=== Final Answer ===")
        typer.echo(result.final_answer)
    else:
        typer.echo("\nNo final answer was produced.", err=True)
        raise typer.Exit(1)
