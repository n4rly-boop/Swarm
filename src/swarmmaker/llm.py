"""LLM client wrapper that enforces structured outputs."""
import copy
import json
import random
import re
import time
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from .schemas import AgentCallMeta, SchemaValidationError, StructuredMode, StructuredParseResult


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response.

    Some models wrap JSON in ```json ... ``` blocks even when asked for raw JSON.
    This function extracts the content inside the fences.
    """
    text = text.strip()
    # Pattern: ```json or ``` at start, ``` at end
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

T = TypeVar("T")


def _make_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Make schema OpenAI-strict compatible.

    - Adds all properties to required array
    - Removes extra keywords from $ref objects (OpenAI doesn't allow them)
    """
    schema = copy.deepcopy(schema)

    def fix_object(obj: Dict[str, Any]) -> None:
        # $ref must be alone - remove other keywords
        if "$ref" in obj:
            keys_to_remove = [k for k in obj.keys() if k != "$ref"]
            for k in keys_to_remove:
                del obj[k]
            return

        if obj.get("type") == "object" and "properties" in obj:
            obj["required"] = list(obj["properties"].keys())
            obj["additionalProperties"] = False
            for prop in obj["properties"].values():
                fix_object(prop)
        elif obj.get("type") == "array" and "items" in obj:
            fix_object(obj["items"])

    # Fix top-level and all $defs
    fix_object(schema)
    for defn in schema.get("$defs", {}).values():
        fix_object(defn)

    return schema


class MetricsTracker:
    """Tracks runtime usage for budgeting and reporting."""

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

    def snapshot(self, steps: int) -> Dict[str, Any]:
        return {
            "elapsed": time.perf_counter() - self.start_time,
            "steps": steps,
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "retries": self.retries,
            "consensus_votes": self.consensus_votes,
            "budget_remaining": max(self.max_total_tokens - self.tokens_total(), 0),
        }

    def increment_retry(self) -> None:
        self.retries += 1

    def add_votes(self, count: int) -> None:
        self.consensus_votes += count

    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMClient:
    """Minimal ChatOpenAI wrapper that always asks for structured JSON."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: int,
        metrics: MetricsTracker,
        structured_mode: StructuredMode,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.timeout = timeout
        self.metrics = metrics
        self.structured_mode = structured_mode
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
        """Enforces structured JSON output matching the provided schema."""
        if self.dry_run:
            payload = self._dry_payload(meta, schema_name)
            text = json.dumps(payload, ensure_ascii=False)
            parsed = parser(payload)
            return StructuredParseResult(content=parsed, raw_text=text)

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is set.")

        # Always enforce structured output format - no fallback to less strict formats
        response_format = self._response_format(schema_name, schema)
        return self._invoke_with_retry(
            messages,
            meta=meta,
            model=model,
            temperature=temperature,
            response_format=response_format,
            parser=parser,
            max_output_tokens=max_output_tokens,
        )

    def _invoke_with_retry(
        self,
        messages: Sequence[BaseMessage],
        *,
        meta: AgentCallMeta,
        model: str,
        temperature: float,
        response_format: Dict[str, Any],
        parser: Callable[[Any], T],
        max_output_tokens: Optional[int],
    ) -> StructuredParseResult[T]:
        """Invokes LLM with structured output enforcement, retry logic, and validation."""
        attempts = 3
        last_error: Optional[Exception] = None
        
        for attempt in range(attempts):
            try:
                # Create LLM instance with enforced structured output format
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    timeout=self.timeout,
                    streaming=False,
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
                    config={"tags": tags, "metadata": metadata},
                )
                
                # Check for API errors in response metadata
                if response.response_metadata.get("error"):
                    error_info = response.response_metadata["error"]
                    error_msg = error_info.get("message", "Unknown API error")
                    if isinstance(error_info.get("metadata"), dict):
                        provider_error = error_info["metadata"].get("raw", "")
                        if provider_error:
                            error_msg = f"{error_msg}: {provider_error[:200]}"
                    raise RuntimeError(f"API error: {error_msg}")
                
                usage = response.response_metadata.get("token_usage", {})
                self.metrics.record_usage(usage)
                
                text = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
                
            except (TypeError, AttributeError) as err:
                # Handle cases where API returns error response with None choices
                if "'NoneType' object is not iterable" in str(err) or "choices" in str(err).lower():
                    err_msg = f"API returned invalid response (likely unsupported response_format). "
                    if self.structured_mode == StructuredMode.json_schema:
                        err_msg += "Try --structured-mode json_object or use a different model."
                    raise RuntimeError(err_msg) from err
                raise
            except Exception as err:  # pragma: no cover - network
                last_error = err
                if attempt == attempts - 1 or not self._should_retry(err):
                    # Provide helpful error message for schema-related failures
                    err_text = str(err).lower()
                    if self.structured_mode == StructuredMode.json_schema and (
                        "schema" in err_text or "response_format" in err_text or "json_schema" in err_text
                    ):
                        raise RuntimeError(
                            "json_schema response_format failed; rerun with --structured-mode json_object "
                            f"or choose a provider that supports schemas. Original error: {err}"
                        ) from err
                    raise
                self.metrics.increment_retry()
                time.sleep(min(2 ** attempt, 4.0))
                continue

            # Strip markdown code fences if present (some models wrap JSON despite instructions)
            text = _strip_code_fences(text)

            # Validate JSON structure
            try:
                data = json.loads(text)
            except json.JSONDecodeError as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            
            # Validate against Pydantic schema
            try:
                parsed = parser(data)
            except ValidationError as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            except Exception as err:
                raise SchemaValidationError(agent=meta.agent, stage=meta.stage, payload=text, error=err)
            
            return StructuredParseResult(content=parsed, raw_text=text)
        
        raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error}")

    def _response_format(self, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self.structured_mode == StructuredMode.json_object:
            return {"type": "json_object"}
        sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in schema_name)[:64] or "SwarmSchema"
        strict_schema = _make_strict_schema(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": sanitized,
                "schema": strict_schema,
                "strict": True,
            },
        }

    def _should_retry(self, err: Exception) -> bool:
        """Determines if an error is transient and worth retrying."""
        text = str(err).lower()
        transient_tokens = ("timeout", "temporarily", "rate limit", "unavailable", "overloaded")
        return any(tok in text for tok in transient_tokens)

    def _dry_payload(self, meta: AgentCallMeta, schema_name: str) -> Dict[str, Any]:
        rand = random.Random(meta.step_id + (meta.voter_id or 0))
        if schema_name == "DecompositionProposal":
            sub_a = f"{meta.stage} sub-problem A (mock)"
            sub_b = f"{meta.stage} sub-problem B (mock)"
            return {
                "subproblem_a": sub_a,
                "subproblem_b": sub_b,
                "compose_fn": "Combine answers sequentially.",
                "is_atomic": rand.random() > 0.5,
                "rationale": "Dry-run decomposition placeholder.",
            }
        if schema_name == "AtomicSolution":
            return {
                "solution": f"mock-solution-{meta.stage}-{meta.step_id}",
                "confidence": round(rand.uniform(0.4, 0.9), 2),
                "work_shown": f"mock work for {meta.stage} / {meta.step_id}",
            }
        return {"mock": f"{meta.agent}-{meta.stage}-{meta.step_id}"}
