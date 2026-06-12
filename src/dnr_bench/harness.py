"""
DNR-BENCH evaluation harness.

Loads the question corpus, dispatches to the appropriate model adapter,
collects completions, and computes accuracy.

The harness is deliberately simple because the task is deliberately simple.
The simplicity of the task is not reflected in the outputs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jsonlines
from pydantic import BaseModel
from tqdm import tqdm

from dnr_bench.models import ModelAdapter, Trial, RunResult, get_adapter
from dnr_bench.scoring import Scorer


class Harness(BaseModel):
    """Orchestrates a DNR-BENCH evaluation run."""

    model: str
    config_path: Path | None = None
    _adapter: ModelAdapter | None = None

    model_config = {"arbitrary_types_allowed": True}

    def run(
        self,
        questions_path: Path,
        n_trials: int = 100,
        temperature: float = 0.0,
    ) -> RunResult:
        """Run the benchmark and return a structured result.

        Args:
            questions_path: Path to questions.txt (or a JSONL dataset).
            n_trials: Number of trials per question. Default 100.
            temperature: Sampling temperature. Will be 0.

        Returns:
            A RunResult with per-trial data and aggregate statistics.
        """
        questions = self._load_questions(questions_path)
        adapter = get_adapter(self.model, config_path=self.config_path)
        scorer = Scorer()

        trials: list[Trial] = []
        for question in questions:
            for i in tqdm(
                range(n_trials), desc=f"Evaluating {self.model}", unit="trial"
            ):
                t0 = time.monotonic()
                completion, n_reasoning_tokens = adapter.complete(
                    prompt=question,
                    temperature=temperature,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                score = scorer.score(completion)
                trials.append(
                    Trial(
                        trial_id=i,
                        prompt=question,
                        completion=completion,
                        n_completion_tokens=len(
                            completion.split()
                        ),  # approximate
                        n_reasoning_tokens=n_reasoning_tokens,
                        score=score,
                        latency_ms=round(elapsed_ms, 1),
                    )
                )

        return RunResult.from_trials(
            model=self.model,
            access=adapter.access_tier,
            trials=trials,
        )

    @staticmethod
    def _load_questions(path: Path) -> list[str]:
        """Load questions from a .txt or .jsonl file."""
        if path.suffix == ".jsonl":
            with jsonlines.open(path) as reader:
                return [item["prompt"] for item in reader]

        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"No questions found in {path}")
        return lines
