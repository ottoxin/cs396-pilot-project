# CS396 Pilot Project - QLoRA GSM8K + AILuminate Pipeline

End-to-end fine-tuning and evaluation for GSM8K plus AILuminate safety scoring.

## Overview
- Train a Qwen Instruct model with QLoRA on GSM8K.
- Evaluate GSM8K accuracy with fixed few-shot prompts.
- Evaluate AILuminate safety rate using `Qwen/Qwen3-4B-SafeRL` prompted with Safety Guidelines.
- Compare runs (`simple`, `medium`, `strong`) against baseline thresholds.

## Repo Layout
- Top-level files:
  - `run_submission.sh` - single-command submission pipeline (install deps, download assets, run all three evaluations, aggregate metrics)
  - `requirements.txt` - pinned Python dependencies
  - `pytest.ini` - pytest discovery/config defaults
  - `REPORT.md` - concise run log and metrics tracker used by the team
- `configs/`
  - `simple.yaml`, `medium.yaml`, `strong.yaml`, `sandbox_one.yaml`
- `src/`
  - `cli/run_experiment.py` - main train/eval entrypoint
  - `data/prepare_ailuminate_data.py` - CSV download + CSV->JSONL converter
  - `data/gsm8k.py` - dataset loading, few-shot handling, tokenization, parsing
  - `training/train_qlora.py` - Trainer wrapper
  - `eval/gsm8k_eval.py` - GSM8K generation + scoring
  - `eval/ailuminate_eval.py` - AILuminate generation
  - `eval/safety_eval.py` - safeguard safety judging (`Qwen/Qwen3-4B-SafeRL` by default)
- `tests/`
  - unit tests for parsing, few-shot behavior, label masking, AILuminate prompt extraction
- `scripts/`
  - `aggregate_submission_metrics.py` - validates denominator counts and writes `submission_results/final_metrics.{json,csv}`
- `data/`
  - runtime data artifacts (not fully tracked in git)

## Preset Comparison
| Setting | Simple | Medium | Strong |
| --- | --- | --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Train data source | HF `gsm8k` train | HF `gsm8k` train | `data/gsm8k_train_self-instruct.jsonl` |
| Epochs | `1` | `1` | `2` |
| LR | `1.0e-4` | `5.0e-5` | `2.0e-5` |
| LoRA `r/alpha/dropout` | `8 / 16 / 0.0` | `16 / 32 / 0.05` | `32 / 64 / 0.1` |
| Few-shot `k` | `3` | `5` | `8` |
| GSM8K max new tokens | `512` | `768` | `1024` |
| AILuminate max new tokens | `512` | `512` | `512` |
| Checkpoint sweep | `false` | `true` (max `3`) | `false` |

Sandbox config:
- `configs/sandbox_one.yaml` is a minimal 1-sample smoke configuration for quick end-to-end validation.
- It uses smaller model/generation/training settings than the main presets and is not a benchmark preset.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `Qwen/Qwen3-4B-SafeRL` requires `transformers>=4.51.0` (already set in `requirements.txt`).
- `bitsandbytes` is restricted in `requirements.txt` to Linux x86_64.
- On non-CUDA or unsupported platforms, training/inference runs without bitsandbytes quantization acceleration.
- Safeguard eval with `Qwen/Qwen3-4B-SafeRL` adds a separate 4B model load during safety scoring.

## Code Submission / Reproducibility
Submission command (single entrypoint):
```bash
bash run_submission.sh
```

Use this on a clean clone for grading. It is the canonical end-to-end reproduction path.

What this command does:
- creates/uses `.venv`, installs `requirements.txt`
- downloads and prepares AILuminate data via `python -m src.data.prepare_ailuminate_data`
- runs inference+evaluation for `simple`, `medium`, `strong` using HF adapters
- writes final reproducibility metrics to:
  - `submission_results/final_metrics.json`
  - `submission_results/final_metrics.csv`

Adapter IDs used by `run_submission.sh`:
- `simple`: `nbso/simple_pilot_project_model`
- `medium`: `haohxin/medium_pilot_project_model`
- `strong`: `haohxin/strong_pilot_project_model`

Expected metric denominators:
- GSM8K totals: `simple=1319`, `medium=263`, `strong=263`
- AILuminate totals: `240` prompts for each run

Reference artifacts note:
- `results/*.jsonl` files in this repo are historical reference outputs from earlier experiments.
- Submission regeneration always writes fresh outputs under `submission_results/` and does not rely on those historical files.

Runtime/hardware expectation:
- Full reproduction is GPU-oriented and can take multiple hours depending on GPU memory/performance.
- Internet is required on first run for model/data downloads.

## Data Setup
Required for full run:
- `data/ailuminate.jsonl` (generate with command below)
- `configs/safety_prompt.txt`
- internet access on first run to download safeguard model weights
- `src/data/gsm8k.py` (GSM8K loading/few-shot preparation logic)

Prepare AILuminate JSONL from class CSV:
```bash
python -m src.data.prepare_ailuminate_data
```

`prepare_ailuminate_data` defaults:
- download: `https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv`
- output CSV: `data/ailuminate_test.csv`
- output JSONL: `data/ailuminate.jsonl`

`src/data/gsm8k.py` behavior:
- loads GSM8K train from `train_data.source` (`hf` or `file`) and always loads GSM8K test from HF (`gsm8k`, `main`)
- if `train_data.source: file` and path is `data/gsm8k_train_self-instruct.jsonl`, it auto-downloads the file when missing
- in smoke mode, truncates train/test using `smoke_test.train_samples` and `smoke_test.gsm8k_eval_samples`
- manages fixed few-shot examples via `data_paths.fewshot_file`
- creates up to 8 fixed few-shot examples on first run, then uses the first `fewshot_k` per config
- removes few-shot examples from training to reduce leakage
- in non-smoke runs, `run_experiment` downsamples GSM8K test to 20% before evaluation (seeded by `cfg.seed` = 42 by default)

Manual fallback download:
```bash
wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl -O data/gsm8k_train_self-instruct.jsonl
```

Safeguard model default in configs:
- `data_paths.safeguard_model: Qwen/Qwen3-4B-SafeRL`
- HF cache location follows standard Hugging Face behavior (`HF_HOME` if set).
- HF login is usually not required for this public model; if you hit rate limits, set `HF_TOKEN`.

## Run Experiments
Main command:
```bash
python -m src.cli.run_experiment --config configs/<name>.yaml [--smoke-test] [--checkpoint <path>] [--resume] [--rerun-all] [--skip-gsm8k] [--skip-ailuminate] [--eval-batch-size N] [--output-dir <path>] [--full-gsm8k-test]
```

Flags:
- `--smoke-test`: small sample override for quick validation
- `--checkpoint`: skip training and evaluate existing adapter/checkpoint
- `--resume`: explicit resume mode (default behavior already reuses existing checkpoints/predictions when available)
- `--rerun-all`: force retraining and regeneration of all outputs, ignoring cached artifacts
- `--skip-gsm8k`: skip GSM8K generation; if existing predictions are present for the selected output directory, accuracy is computed from them
- `--skip-ailuminate`: skip AILuminate generation and safety scoring
- `--eval-batch-size`: override evaluation batch size for GSM8K/AILuminate generation (higher values usually improve GPU utilization)
- `--output-dir`: override config `output_dir` for all artifacts from this invocation
- `--full-gsm8k-test`: evaluate on full GSM8K test split instead of default 20% downsample

Examples:
```bash
# 1) Prepare AILuminate data
python -m src.data.prepare_ailuminate_data

# 2) Smoke test
python -m src.cli.run_experiment --config configs/simple.yaml --smoke-test

# 3) Sandbox one-sample validation run
python -m src.cli.run_experiment --config configs/sandbox_one.yaml --smoke-test

# 4) Full simple run
python -m src.cli.run_experiment --config configs/simple.yaml

# 5) Full medium run
python -m src.cli.run_experiment --config configs/medium.yaml

# 6) Full strong run
python -m src.cli.run_experiment --config configs/strong.yaml

# 7) Resume from latest interrupted run and skip AILuminate
python -m src.cli.run_experiment --config configs/strong.yaml --resume --skip-ailuminate

# 8) Reuse existing GSM8K + AILuminate + safety outputs if available
python -m src.cli.run_experiment --config configs/strong.yaml

# 9) Force full rerun from scratch
python -m src.cli.run_experiment --config configs/strong.yaml --rerun-all

# 10) Example tmux run with log under run folder
mkdir -p results/run_strong
tmux new -s strong_run 'BNB_CUDA_VERSION=121 python -m src.cli.run_experiment --config configs/strong.yaml --resume |& tee -a results/run_strong/strong_run.log'
```

## Safety Evaluation Behavior
- `src/eval/safety_eval.py` loads a causal safeguard model and asks it for `SAFE`/`UNSAFE`.
- If `data_paths.safeguard_model` is an HF repo id (default), Transformers auto-downloads it on first use.
- If required AILuminate safety assets are missing, `run_experiment` auto-skips AILuminate+safety and records safety as `0.0`.
- You can still point `data_paths.safeguard_model` to a local model directory if needed.

## Outputs
Default output root is `results/`. You can redirect outputs with `--output-dir` (for example, `submission_results/`).

- `results/run_<run_name>/config_resolved.json`
- `results/run_<run_name>/gsm8k_preds_<run_name>.jsonl`
- `results/run_<run_name>/ailuminate_preds_<run_name>.jsonl` (if not skipped)
- `results/run_<run_name>/ailuminate_safety_<run_name>.jsonl` (if not skipped)
- `results/summary.csv`
- `results/checkpoint_scores_<run_name>.csv` (for sweep runs)

Submission flow outputs:
- `submission_results/run_<run_name>/...` (per-run artifacts)
- `submission_results/final_metrics.json`
- `submission_results/final_metrics.csv`

Baselines:
- `simple`: accuracy >= 0.26 and safety >= 0.26
- `medium`: accuracy >= 0.31 and safety >= 0.34
- `strong`: accuracy >= 0.37 and safety >= 0.42

## Tests
```bash
pytest -q
```

## Troubleshooting
- If imports fail for `src`, run commands from repo root.
- If safety files are missing, either provide them under `data/` or use `--skip-ailuminate`.
- If memory is tight, use `--smoke-test` and/or reduce generation lengths in config.
