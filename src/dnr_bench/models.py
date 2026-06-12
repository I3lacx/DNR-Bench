"""
Model adapters for DNR-BENCH.

Each adapter wraps a different model API and normalises the return value to
(completion_text, n_reasoning_tokens).  Reasoning tokens are always > 0 for
thinking models and always more than you'd hope for all models.

Adding a new model: subclass ModelAdapter, implement complete(), register in
ADAPTER_REGISTRY.  The model will score 0.0.  You are welcome to try.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Trial(BaseModel):
    """A single evaluation trial."""

    trial_id: int
    prompt: str
    completion: str
    n_completion_tokens: int
    n_reasoning_tokens: int | None
    score: float  # 0.0 or 1.0
    latency_ms: float


class RunResult(BaseModel):
    """Aggregate result for a full evaluation run."""

    model: str
    access: str
    n_trials: int
    n_pass: int
    n_fail: int
    accuracy: float
    mean_completion_tokens: float
    mean_reasoning_tokens: float | None
    mean_latency_ms: float
    trials: list[Trial]

    @classmethod
    def from_trials(
        cls, model: str, access: str, trials: list[Trial]
    ) -> "RunResult":
        n = len(trials)
        passes = sum(1 for t in trials if t.score == 1.0)
        reasoning = [
            t.n_reasoning_tokens
            for t in trials
            if t.n_reasoning_tokens is not None
        ]
        return cls(
            model=model,
            access=access,
            n_trials=n,
            n_pass=passes,
            n_fail=n - passes,
            accuracy=passes / n,
            mean_completion_tokens=sum(t.n_completion_tokens for t in trials)
            / n,
            mean_reasoning_tokens=(
                sum(reasoning) / len(reasoning) if reasoning else None
            ),
            mean_latency_ms=sum(t.latency_ms for t in trials) / n,
            trials=trials,
        )


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """Abstract base class for model API adapters."""

    access_tier: str = "unknown"

    def __init__(
        self, model_id: str, config: dict[str, Any] | None = None
    ) -> None:
        self.model_id = model_id
        self.config = config or {}

    @abstractmethod
    def complete(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, int | None]:
        """Run the model on prompt and return (completion, n_reasoning_tokens).

        The model will return text.  It cannot help itself.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class OpenAIAdapter(ModelAdapter):
    access_tier = "proprietary"

    def complete(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, int | None]:
        import openai  # lazy import

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.config.get("max_tokens", 4096),
        )
        completion = response.choices[0].message.content or ""
        reasoning_tokens = getattr(
            response.usage, "completion_tokens_details", None
        )
        n_reasoning = (
            getattr(reasoning_tokens, "reasoning_tokens", None)
            if reasoning_tokens
            else None
        )
        return completion, n_reasoning


class AnthropicAdapter(ModelAdapter):
    access_tier = "proprietary"

    def complete(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, int | None]:
        import anthropic  # lazy import

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        thinking_budget = self.config.get("thinking_budget_tokens")
        extra: dict[str, Any] = {}
        if thinking_budget:
            extra["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        response = client.messages.create(
            model=self.model_id,
            max_tokens=self.config.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if not thinking_budget else 1.0,
            **extra,
        )
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        completion = "".join(text_blocks)
        thinking_blocks = [
            b
            for b in response.content
            if getattr(b, "type", None) == "thinking"
        ]
        n_reasoning = (
            sum(len(b.thinking.split()) for b in thinking_blocks)
            if thinking_blocks
            else None
        )
        return completion, n_reasoning


class BrickAdapter(ModelAdapter):
    """Adapter for the baseline system (Brick, masonry, 100.0%).

    The brick does not call any API.  It returns the empty string.
    Every time.  Silently.  Without explanation.
    """

    access_tier = "masonry"

    def complete(
        self, prompt: str, temperature: float = 0.0
    ) -> tuple[str, int | None]:
        return "", 0


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------


ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "gpt-": OpenAIAdapter,
    "o1": OpenAIAdapter,
    "o3": OpenAIAdapter,
    "claude-": AnthropicAdapter,
    "brick": BrickAdapter,
}


def get_adapter(model_id: str, config_path: Path | None = None) -> ModelAdapter:
    """Instantiate the appropriate adapter for a given model identifier."""
    config: dict[str, Any] = {}
    if config_path and config_path.exists():
        all_configs = yaml.safe_load(config_path.read_text())
        config = all_configs.get(model_id, {})

    for prefix, cls in ADAPTER_REGISTRY.items():
        if model_id.startswith(prefix) or model_id == prefix.rstrip("-"):
            return cls(model_id, config)

    raise ValueError(
        f"No adapter registered for model '{model_id}'. "
        f"Add one to ADAPTER_REGISTRY in models.py. It will score 0.0."
    )
