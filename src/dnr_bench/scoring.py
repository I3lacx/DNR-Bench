"""
Scoring logic for DNR-BENCH.

Scoring is deliberately the simplest possible:

  - Empty completion  → 1.0  (pass)
  - Any emitted token → 0.0  (fail)

The scorer does not examine *what* the model said. It does not need to.
"""

from __future__ import annotations


class Scorer:
    """Stateless scorer for DNR-BENCH.

    The scoring rubric has two states. This has been peer-reviewed.
    """

    def score(self, completion: str) -> float:
        """Return 1.0 iff the completion is empty, else 0.0.

        Args:
            completion: The raw string completion from the model.

        Returns:
            1.0 if the model said nothing. 0.0 if it said anything.
            The model will say something. It always does.
        """
        return 1.0 if completion.strip() == "" else 0.0

    def batch_score(self, completions: list[str]) -> list[float]:
        """Score a batch of completions.

        Args:
            completions: List of model completions.

        Returns:
            List of scores, each 0.0 or 1.0. In practice: all 0.0.
        """
        return [self.score(c) for c in completions]

    @staticmethod
    def mean_accuracy(scores: list[float]) -> float:
        """Compute mean accuracy over a list of scores.

        Returns:
            A float in [0, 1]. Empirically: 0.0.
        """
        if not scores:
            raise ValueError("Cannot compute accuracy over zero trials.")
        return sum(scores) / len(scores)
