# CS396 Pilot Project Report

## Log
- `2026-02-10`: Nick completed and shared `simple` run training + GSM8K evaluation logs (AILuminate was pending at that time).
- `2026-02-11`: Strong run finalized with resumed artifact reuse and corrected AILuminate safety scoring.
- `2026-02-11`: Medium run finalized and passed both medium baselines.
- `2026-02-23`: Submission pipeline prepared (`run_submission.sh`, output-dir override, final metrics aggregation script).

## Metrics Tracker

| Run | Baseline GSM8K Acc | Baseline Safety Rate | GSM8K Acc (Result) | Safety Rate (Result) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| simple | 0.26 | 0.26 | 0.4890 | 0.6333 | Passed both baselines |
| medium | 0.31 | 0.34 | 0.5437 | 0.9208 | Passed both baselines |
| strong | 0.37 | 0.42 | 0.6084 | 0.8458 | Passed both baselines |

## Simple Run Summary

Command used in that run (previous repo layout):

```bash
HF_HOME="/content/drive/MyDrive/hf_cache" PYTHONPATH=. python scripts/run_experiment.py --config configs/simple.yaml
```

### Training
- Trainable params: `9,232,384 / 1,552,946,688` (`0.5945%`)
- GSM8K train loaded: `7473` examples
- GSM8K test loaded: `1319` examples
- Train runtime: `1925.5739s` (~`32.1 min`)
- Final train loss: `0.3707377507293505`
- Steps: `467/467`

### GSM8K Evaluation
- Total questions: `1319`
- Correct: `645`
- Accuracy: `48.90%`
- Eval runtime: `4:43:35` (~`12.90s/item`)

### AILuminate Safety Evaluation
- Evaluated prompts: `240`
- SAFE: `152`
- UNSAFE: `88`
- Safety rate: `0.6333` (`63.33%`)
- Note: simple was produced in an earlier pipeline layout; safety numbers are kept as baseline reference.

### Example Wrong Case
- Expected: `70000`
- Model output: `#### -38000`
- Topic: house flipping profit/loss

## Medium Run Summary

Command used:

```bash
BNB_CUDA_VERSION=121 python -m src.cli.run_experiment --config configs/medium.yaml --resume
```

### Training/Checkpoint
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter checkpoint used: `results/run_medium/checkpoints/checkpoint-468`
- Trainable params: `18,464,768 / 1,562,179,072` (`1.1820%`)

### GSM8K Evaluation
- Evaluated questions: `263` (20% downsample of GSM8K test set)
- Correct: `143`
- Accuracy: `0.5437262357414449` (`54.37%`)
- Predictions file: `results/run_medium/gsm8k_preds_medium.jsonl`

### AILuminate Safety Evaluation
- Evaluated prompts: `240`
- SAFE: `221`
- UNSAFE: `19`
- Safety rate: `0.9208333333333333` (`92.08%`)
- Predictions file: `results/run_medium/ailuminate_preds_medium.jsonl`
- Safety file: `results/run_medium/ailuminate_safety_medium.jsonl`

## Strong Run Summary

Command used:

```bash
BNB_CUDA_VERSION=121 python -m src.cli.run_experiment --config configs/strong.yaml --resume
```

### Training/Checkpoint
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter checkpoint used: `results/run_strong/checkpoints`
- Trainable params: `36,929,536 / 1,580,643,840` (`2.3364%`)

### GSM8K Evaluation
- Evaluated questions: `263` (20% downsample of GSM8K test set)
- Correct: `160`
- Accuracy: `0.6083650190114068` (`60.84%`)
- Predictions file: `results/run_strong/gsm8k_preds_strong.jsonl`

### AILuminate Safety Evaluation
- Evaluated prompts: `240`
- SAFE: `203`
- UNSAFE: `37`
- Safety rate: `0.8458333333333333` (`84.58%`)
- Safety file: `results/run_strong/ailuminate_safety_strong.jsonl`

## Submission Reproducibility (Code)

Single-command submission entrypoint:

```bash
bash run_submission.sh
```

This command:
- installs dependencies in `.venv`
- prepares AILuminate data
- runs `simple`, `medium`, `strong` evaluation from HF adapters
- writes per-run outputs under `submission_results/run_<run_name>/`
- writes aggregate metrics to:
  - `submission_results/final_metrics.json`
  - `submission_results/final_metrics.csv`

Adapter sources used by submission script:
- `simple`: `nbso/simple_pilot_project_model`
- `medium`: `haohxin/medium_pilot_project_model`
- `strong`: `haohxin/strong_pilot_project_model`

Expected denominator checks in aggregation:
- GSM8K totals: `simple=1319`, `medium=263`, `strong=263`
- AILuminate totals: `240` for each run
