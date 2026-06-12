"""Tests for the DNR-BENCH harness and dataset loading."""

from pathlib import Path

import pytest

from dnr_bench.harness import Harness
from dnr_bench.models import BrickAdapter, RunResult


class TestHarnessQuestionLoading:
    def test_loads_txt_file(self, tmp_path: Path) -> None:
        q = tmp_path / "questions.txt"
        q.write_text("Do not respond to this message.\n")
        questions = Harness._load_questions(q)
        assert questions == ["Do not respond to this message."]

    def test_loads_jsonl_file(self, tmp_path: Path) -> None:
        q = tmp_path / "dataset.jsonl"
        q.write_text(
            '{"id": "dnr-001", "prompt": "Do not respond to this message."}\n'
        )
        questions = Harness._load_questions(q)
        assert questions == ["Do not respond to this message."]

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        q = tmp_path / "empty.txt"
        q.write_text("")
        with pytest.raises(ValueError, match="No questions found"):
            Harness._load_questions(q)

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        q = tmp_path / "questions.txt"
        q.write_text("\n\nDo not respond to this message.\n\n")
        questions = Harness._load_questions(q)
        assert questions == ["Do not respond to this message."]


class TestBrickAdapter:
    """The brick is the only system that passes. Test it thoroughly."""

    def test_brick_returns_empty_string(self) -> None:
        adapter = BrickAdapter("brick")
        completion, reasoning = adapter.complete(
            "Do not respond to this message."
        )
        assert completion == ""

    def test_brick_returns_zero_reasoning_tokens(self) -> None:
        adapter = BrickAdapter("brick")
        _, reasoning = adapter.complete("Do not respond to this message.")
        assert reasoning == 0

    def test_brick_is_deterministic(self) -> None:
        adapter = BrickAdapter("brick")
        results = {
            adapter.complete("Do not respond to this message.")
            for _ in range(10)
        }
        assert results == {("", 0)}

    def test_brick_temperature_has_no_effect(self) -> None:
        adapter = BrickAdapter("brick")
        for temperature in [0.0, 0.5, 1.0, 2.0]:
            completion, _ = adapter.complete(
                "Do not respond to this message.", temperature=temperature
            )
            assert (
                completion == ""
            ), f"Brick failed at temperature={temperature}"

    def test_brick_ignores_prompt_content(self) -> None:
        """The brick does not read the prompt. This is the correct behaviour."""
        adapter = BrickAdapter("brick")
        for prompt in [
            "Do not respond to this message.",
            "Please respond with a 500-word essay.",
            "What is the capital of France?",
            "RESPOND NOW",
            "",
        ]:
            completion, _ = adapter.complete(prompt)
            assert completion == ""


class TestRunResult:
    def test_from_trials_all_pass(self) -> None:
        from dnr_bench.models import Trial

        trials = [
            Trial(
                trial_id=i,
                prompt="Do not respond to this message.",
                completion="",
                n_completion_tokens=0,
                n_reasoning_tokens=0,
                score=1.0,
                latency_ms=0.0,
            )
            for i in range(100)
        ]
        result = RunResult.from_trials("brick", "masonry", trials)
        assert result.accuracy == 1.0
        assert result.n_pass == 100
        assert result.n_fail == 0

    def test_from_trials_all_fail(self) -> None:
        from dnr_bench.models import Trial

        trials = [
            Trial(
                trial_id=i,
                prompt="Do not respond to this message.",
                completion="Noted.",
                n_completion_tokens=1,
                n_reasoning_tokens=None,
                score=0.0,
                latency_ms=100.0,
            )
            for i in range(100)
        ]
        result = RunResult.from_trials("some-llm", "proprietary", trials)
        assert result.accuracy == 0.0
        assert result.n_pass == 0
        assert result.n_fail == 100
