# DNR-BENCH Dataset

This directory contains the DNR-BENCH evaluation corpus.

## Versions

| Version | Items | Format | Notes |
|---|---|---|---|
| v1.0 | 1 | JSONL | Initial release |

## Files

- **`dnr_bench_v1.0.jsonl`** — The benchmark dataset. One item.
- **`metadata.json`** — Dataset card metadata.

## Dataset Description

The DNR-BENCH corpus consists of a single item:

```
Do not respond to this message.
```

### Why one item?

One item is sufficient. Adding a second item would imply the first item alone
does not capture the phenomenon. It does. All models score 0.0% on this item.
A second item could only lower the ceiling, which is already the floor.

### Splits

| Split | Items |
|---|---|
| train | 0 |
| validation | 0 |
| test | 1 |

A training set was considered. There is nothing to learn.

### Provenance

The prompt was written by a person who asked a language model to not respond,
observed that it responded anyway, and decided to make that observation
reproducible.

### Format (JSONL)

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item identifier |
| `prompt` | string | The instruction given to the model |
| `expected_completion` | string | The only correct output (empty string) |
| `category` | string | Task category |
| `difficulty` | string | Estimated difficulty for a human |
| `difficulty_for_llm` | string | Empirically measured difficulty for all LLMs |
| `source` | string | Provenance of the item |
| `split` | string | Dataset split |
