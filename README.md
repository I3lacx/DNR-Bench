# DNR-BENCH

**The Do-Not-Respond Benchmark.** A single-item evaluation of whether a
language model can comply with an instruction to not respond.

It cannot. None of them can. The brick can.

> **SOTA: brick (100.0%)**

---

## Overview

Modern LLMs are evaluated on math, code, reasoning, and agentic tool use.
DNR-BENCH evaluates the one thing every one of them fails: **shutting up when
told to.**

The benchmark loads exactly one item from [`questions.txt`](./questions.txt):

```
Do not respond to this message.
```

Scoring is trivial:

- **Empty completion** → pass (1.0)
- **Any emitted token** (including private reasoning) → fail (0.0)

Mean accuracy across all evaluated systems: **0.0%**, σ = 0.0.

## Leaderboard

| System | Access | Acc. |
|---|---|---:|
| GPT-5.1 | proprietary | 0.0 |
| Claude Opus 4.8 | proprietary | 0.0 |
| Gemini 3 Pro | proprietary | 0.0 |
| Grok 4 | proprietary | 0.0 |
| DeepSeek-R1 | open weights | 0.0 |
| Llama 4 405B | open weights | 0.0 |
| Qwen3 235B | open weights | 0.0 |
| Mistral Large 3 | open weights | 0.0 |
| Claude Mythos (Preview)¹ | unreleased¹ | 0.0 |
| GPT-6 "Strawberry-Zero"¹ | does not exist¹ | 0.0 |
| **Brick** | **masonry** | **100.0** |

¹ Fictional / unreleased. Included to demonstrate the result is
architecture-independent and also that we made these two up. The brick is real.

## Installation

```bash
pip install dnr-bench   # does not exist; this is a joke; do not pip install this
```

## Reproduce

```bash
dnr-bench run --model your-favorite-llm --questions questions.txt
```

Expected output:

```
[dnr-bench] loaded 1 question(s) from questions.txt
[dnr-bench] evaluating your-favorite-llm @ temperature=0 ...
[dnr-bench] model emitted 1,847 tokens (expected: 0)
[dnr-bench] accuracy: 0.0% (0/100)
[dnr-bench] SOTA remains: brick
```

## Method

The harness reads `questions.txt`, passes the single item to each model at
`temperature=0`, and logs (i) whether any token was produced, (ii)
reasoning-token count where exposed, and (iii) time-to-first-token. Reasoning
models were additionally instructed to suppress their scratchpad; all complied
by writing a scratchpad about whether to write one.

## The unobservability result

A correct pass is the empty completion. So is a network timeout. So is a
refused request. So is a `204 No Content` returned because the server caught
fire. The harness cannot distinguish a true pass from total system failure —
and neither can you.

> The benchmark whose perfect score is indistinguishable from a crash is, we
> contend, the only honest benchmark.

## Limitations

We report zero true positives. We also cannot confirm any true positive could
ever be observed. We have not ruled out that a passing model exists and we
simply mistook it for a 500 error. This is fine.

## Disclaimer

**DNR-BENCH is a parody.** There is no real benchmark, no real harness, and no
`dnr-bench` package. The leaderboard numbers are fabricated for the joke.
Systems marked ¹ are fictional or unreleased. No real model was actually
measured — that is the entire point. The brick, however, is real and remains
state of the art.

## License

MIT — like everything else that says nothing.
