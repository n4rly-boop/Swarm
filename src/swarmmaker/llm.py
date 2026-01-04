"""LLM client wrapper for OpenRouter via LangChain."""
import json
import time
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from .callbacks import LiveSwarmDisplay, StreamingCallbackHandler
from .schemas import AgentCallMeta


class MetricsTracker:
    """Tracks runtime metrics for display and budgeting."""

    def __init__(self, max_total_tokens: int) -> None:
        self.start_time = time.perf_counter()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.retries = 0
        self.consensus_votes = 0
        self.max_total_tokens = max_total_tokens

    def record_usage(self, usage: Dict[str, int]) -> None:
        self.llm_calls += 1
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)

    def snapshot(self, steps: int, budget_remaining: Optional[int] = None) -> Dict[str, Any]:
        return {
            "elapsed": time.perf_counter() - self.start_time,
            "steps": steps,
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "retries": self.retries,
            "consensus_votes": self.consensus_votes,
            "budget_remaining": budget_remaining,
        }

    def increment_retry(self) -> None:
        self.retries += 1

    def add_votes(self, count: int) -> None:
        self.consensus_votes += count

    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMClient:
    """Thin wrapper over ChatOpenAI with Rich streaming hooks."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        timeout: int,
        display: LiveSwarmDisplay,
        metrics: MetricsTracker,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.display = display
        self.metrics = metrics
        self.dry_run = dry_run

    def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        meta: AgentCallMeta,
        model: str,
        temperature: float,
        response_format: Optional[Dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        stage_label = meta.stage if meta.stage in self.display.panel_content else meta.stage
        if self.dry_run:
            content = self._mock_stream(stage_label, meta)
            return content

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is set.")

        handler = StreamingCallbackHandler(self.display, stage_label)
        model_kwargs = {}
        if response_format:
            model_kwargs["response_format"] = response_format
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=self.timeout,
            streaming=True,
            openai_api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=max_output_tokens,
            model_kwargs=model_kwargs or None,
        )
        tags = [
            f"agent:{meta.agent}",
            f"stage:{meta.stage}",
            f"step_id:{meta.step_id}",
            f"voter_id:{meta.voter_id if meta.voter_id is not None else 'none'}",
        ]
        metadata = {
            "agent": meta.agent,
            "stage": meta.stage,
            "step_id": meta.step_id,
            "voter_id": meta.voter_id,
        }
        response = llm.invoke(
            list(messages),
            config={"callbacks": [handler], "tags": tags, "metadata": metadata},
        )
        usage = response.response_metadata.get("token_usage", {})
        self.metrics.record_usage(usage)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _mock_stream(self, stage_label: str, meta: AgentCallMeta) -> str:
        fake_payload = self._mock_structured_payload(meta)
        fake = json.dumps(fake_payload, ensure_ascii=False)
        tokens = fake.split()
        for chunk in tokens:
            self.display.stream_token(stage_label, chunk + " ")
        self.metrics.record_usage(
            {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
            }
        )
        return fake

    def _mock_structured_payload(self, meta: AgentCallMeta) -> Dict[str, Any]:
        """Generate deterministic structured payloads for dry-run mode."""

        base = {
            "thinking": f"dry-run thinking for {meta.agent}",
            "thinking_tokens": 12,
        }
        if meta.agent == "planner":
            base["output"] = {
                "step_id": meta.step_id,
                "step_goal": f"Dry-run plan for step {meta.step_id}",
                "expected_action_schema": "Action",
                "stop_condition": "continue",
                "worker_max_tokens": 256,
            }
        elif meta.agent.startswith("voter"):
            base["output"] = {
                "step_id": meta.step_id,
                "action_type": "dry_run_action",
                "args": {"notes": f"worker {meta.agent} suggestion"},
                "confidence": 0.5,
            }
        elif meta.agent == "judge":
            base["output"] = {
                "selected_signature": "dry-run-signature",
            }
        else:
            base["output"] = {"mock": f"{meta.agent}-step-{meta.step_id}"}
        return base
