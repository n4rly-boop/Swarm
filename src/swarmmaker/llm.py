"""LLM client wrapper with provider-level structured output."""
import json
import time
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from .callbacks import LiveSwarmDisplay, StreamingCallbackHandler
from .schemas import (
    AgentCallMeta,
    SchemaValidationError,
    StructuredMode,
    StructuredParseResult,
)


T = TypeVar("T")


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
    """Thin wrapper over ChatOpenAI that always requests structured responses."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: int,
        display: LiveSwarmDisplay,
        metrics: MetricsTracker,
        structured_mode: StructuredMode,
        stream: bool,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.timeout = timeout
        self.display = display
        self.metrics = metrics
        self.structured_mode = structured_mode
        self.stream = stream
        self.dry_run = dry_run

    def structured_completion(
        self,
        messages: Sequence[BaseMessage],
        *,
        meta: AgentCallMeta,
        model: str,
        temperature: float,
        schema_name: str,
        schema: Dict[str, Any],
        parser: Callable[[Any], T],
        max_output_tokens: Optional[int] = None,
    ) -> StructuredParseResult[T]:
        if self.dry_run:
            payload = self._dry_payload(meta)
            text = json.dumps(payload, ensure_ascii=False)
            self._stream_fake(text, meta.stage)
            parsed = parser(payload)
            return StructuredParseResult(content=parsed, raw_text=text)

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is set.")

        response_format = self._response_format(schema_name, schema)
        attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                text = self._invoke(
                    messages,
                    meta=meta,
                    model=model,
                    temperature=temperature,
                    response_format=response_format,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as err:  # pragma: no cover - network heavy
                last_error = err
                if attempt == attempts - 1 or not self._should_retry(err):
                    if self.structured_mode == StructuredMode.json_schema and self._looks_like_schema_error(err):
                        raise RuntimeError(
                            "json_schema response_format failed; switch to --structured-mode json_object "
                            f"or choose a provider that supports schemas. Original error: {err}"
                        ) from err
                    raise
                self.metrics.increment_retry()
                time.sleep(min(2 ** attempt, 4.0))
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            try:
                parsed = parser(data)
            except ValidationError as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            except Exception as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            return StructuredParseResult(content=parsed, raw_text=text)
        raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error}")

    # Internal helpers -------------------------------------------------
    def _invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        meta: AgentCallMeta,
        model: str,
        temperature: float,
        response_format: Dict[str, Any],
        max_output_tokens: Optional[int],
    ) -> str:
        handler = StreamingCallbackHandler(self.display, meta.stage)
        callbacks = [handler] if self.stream else []
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=self.timeout,
            streaming=self.stream,
            openai_api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=max_output_tokens,
            model_kwargs={"response_format": response_format},
        )
        tags = [
            f"agent:{meta.agent}",
            f"stage:{meta.stage}",
            f"step_id:{meta.step_id}",
        ]
        if meta.voter_id is not None:
            tags.append(f"voter_id:{meta.voter_id}")
        metadata = {
            "agent": meta.agent,
            "stage": meta.stage,
            "step_id": meta.step_id,
            "voter_id": meta.voter_id,
        }
        response = llm.invoke(
            list(messages),
            config={"callbacks": callbacks, "tags": tags, "metadata": metadata},
        )
        usage = response.response_metadata.get("token_usage", {})
        self.metrics.record_usage(usage)
        return response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)

    def _response_format(self, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self.structured_mode == StructuredMode.json_object:
            return {"type": "json_object"}
        sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in schema_name)[:64] or "SwarmSchema"
        return {
            "type": "json_schema",
            "json_schema": {
                "name": sanitized,
                "schema": schema,
                "strict": True,
            },
        }

    def _should_retry(self, err: Exception) -> bool:
        text = str(err).lower()
        transient_tokens = ("timeout", "temporarily", "rate limit", "unavailable", "overloaded")
        return any(tok in text for tok in transient_tokens)

    def _looks_like_schema_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return "schema" in text or "response_format" in text or "json_schema" in text

    def _dry_payload(self, meta: AgentCallMeta) -> Dict[str, Any]:
        if meta.agent == "planner":
            return {
                "step_id": meta.step_id,
                "step_goal": f"draft final answer for step {meta.step_id}",
                "stop_condition": "continue",
                "worker_max_tokens": 256,
            }
        if meta.agent.startswith("voter"):
            return {
                "step_id": meta.step_id,
                "action_type": "NOTE",
                "args": {"content": f"dry-run note from {meta.agent}"},
                "rationale": None,
                "confidence": 0.5,
            }
        if meta.agent == "judge":
            return {"selected_signature": "none"}
        return {"mock": f"{meta.agent}-{meta.stage}-{meta.step_id}"}

    def _stream_fake(self, text: str, stage: str) -> None:
        tokens = text.split()
        for token in tokens:
            self.display.stream_token(stage, token + " ")
        self.metrics.record_usage({"prompt_tokens": 0, "completion_tokens": len(tokens)})
