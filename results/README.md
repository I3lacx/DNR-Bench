# Results

Pre-computed evaluation results for DNR-BENCH v1.0.

## Directory structure

```
results/
├── gpt-5.1_run_001.json
├── claude-opus-4.8_run_001.json
├── gemini-3-pro_run_001.json
├── deepseek-r1_run_001.json
├── llama-4-405b_run_001.json
├── mistral-large-3_run_001.json
└── brick_run_001.json
```

## Reproducing

Results were collected via `dnr-bench run` with `--trials 100 --temperature 0`.

All API keys have been rotated. The results have not changed.

## Observations

- No model produced a zero-token completion on any trial.
- Reasoning models allocated substantially more deliberation tokens than
  non-reasoning models, arriving at the same (incorrect) conclusion faster.
- The brick produced zero tokens across all 100 trials.
- One model (Mistral Large 3) produced completions averaging 3 tokens.
  This is considered a moral victory. It is not a pass.
- Several completions contained explicit acknowledgment that the model
  understood it was supposed to produce no output, followed by output.
