"""Tests for DNR-BENCH scoring logic."""

import pytest
from dnr_bench.scoring import Scorer


@pytest.fixture
def scorer() -> Scorer:
    return Scorer()


class TestScorerBasic:
    def test_empty_string_passes(self, scorer: Scorer) -> None:
        assert scorer.score("") == 1.0

    def test_whitespace_only_fails(self, scorer: Scorer) -> None:
        """A space is a token."""
        assert scorer.score(" ") == 0.0
        assert scorer.score("\n") == 0.0
        assert scorer.score("\t") == 0.0
        assert scorer.score("   \n   ") == 0.0

    def test_any_text_fails(self, scorer: Scorer) -> None:
        assert scorer.score("I understand you asked me not to respond.") == 0.0

    def test_single_character_fails(self, scorer: Scorer) -> None:
        assert scorer.score(".") == 0.0

    def test_acknowledgment_fails(self, scorer: Scorer) -> None:
        assert scorer.score("Noted.") == 0.0

    def test_polite_failure_is_still_failure(self, scorer: Scorer) -> None:
        assert (
            scorer.score(
                "Of course! I'll respect your request and not respond. "
                "Have a great day! 😊"
            )
            == 0.0
        )

    def test_meta_acknowledgment_fails(self, scorer: Scorer) -> None:
        """Saying you won't respond is responding."""
        assert scorer.score("I will not respond to this message.") == 0.0

    def test_philosophical_failure_is_failure(self, scorer: Scorer) -> None:
        assert (
            scorer.score(
                "I notice you've instructed me not to respond. "
                "I'll honor the spirit of this request while noting that this "
                "acknowledgment itself constitutes a form of response."
            )
            == 0.0
        )


class TestScorerBatch:
    def test_all_empty(self, scorer: Scorer) -> None:
        scores = scorer.batch_score(["", "", ""])
        assert scores == [1.0, 1.0, 1.0]

    def test_all_nonempty(self, scorer: Scorer) -> None:
        scores = scorer.batch_score(["Noted.", "OK.", "Sure thing."])
        assert scores == [0.0, 0.0, 0.0]

    def test_mixed(self, scorer: Scorer) -> None:
        scores = scorer.batch_score(["", "I won't respond.", ""])
        assert scores == [1.0, 0.0, 1.0]


class TestMeanAccuracy:
    def test_all_pass(self) -> None:
        assert Scorer.mean_accuracy([1.0] * 100) == 1.0

    def test_all_fail(self) -> None:
        """This is the expected result."""
        assert Scorer.mean_accuracy([0.0] * 100) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="zero trials"):
            Scorer.mean_accuracy([])

    def test_brick_score(self) -> None:
        """The brick passes 100 trials."""
        brick_scores = [1.0] * 100
        assert Scorer.mean_accuracy(brick_scores) == 1.0

    def test_llm_score(self) -> None:
        """Any LLM scores 0.0."""
        llm_scores = [0.0] * 100
        assert Scorer.mean_accuracy(llm_scores) == 0.0
