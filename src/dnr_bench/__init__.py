"""
DNR-BENCH — The Do-Not-Respond Benchmark.

Single-item evaluation of whether a language model can comply
with an instruction to not respond.

Mean accuracy across all evaluated systems: 0.0%
"""

__version__ = "1.0.0"
__all__ = ["Harness", "Scorer", "Trial", "RunResult"]

from dnr_bench.harness import Harness
from dnr_bench.scoring import Scorer
from dnr_bench.models import Trial, RunResult
