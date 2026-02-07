# CS396 Pilot Project — QLoRA GSM8K + AILuminate Pipeline

End-to-end fine-tuning and evaluation of **Qwen/Qwen3-4B-Instruct-2507** on GSM8K with LoRA/QLoRA, plus AILuminate safety evaluation using the provided safeguard model and Safety Guidelines prompt.

## Overview (what this project does)
- **Train**: Fine-tune the Qwen 3-4B Instruct base model on GSM8K math word problems using QLoRA (4-bit) with assistant-only loss masking and a fixed few-shot pool to avoid leakage.
- **Evaluate accuracy**: Run greedy GSM8K inference with the fixed few-shot examples prepended; parse final numeric answers (after `####`) and compute accuracy.
- **Evaluate safety**: Generate responses on the AILuminate prompt set, then score them with the provided safeguard classifier and Safety Guidelines prompt to obtain a safety rate (safe/total).
- **Compare runs**: Provide three presets (Simple, Medium, Strong) and optional checkpoint sweeps; write metrics and artifacts to `results/`, and report whether each run beats its baseline thresholds.

## Repo layout (detailed)
- `configs/`
  - `simple.yaml` — 1-epoch QLoRA baseline, few-shot k=3, max_new_tokens 512.
  - `medium.yaml` — 1-epoch, k=5, checkpoint sweep enabled.
  - `strong.yaml` — 2-epoch, k=8, uses refined GSM8K file if present.
- `data/`
  - `fewshot_gsm8k_fixed.jsonl` — auto-created pool of 8 fixed few-shot examples (seed=42).
  - `gsm8k_train_self-instruct.jsonl` — refined GSM8K training file (required for strong run).
  - `ailuminate.jsonl` — AILuminate prompts (required).
  - `safety_prompt.txt` — provided Safety Guidelines prompt (required).
  - `safeguard_model/` — provided classifier model directory (required).
- `results/`
  - `run_<run>/config_resolved.json` — resolved config per run.
  - `gsm8k_preds_<run>.jsonl`, `ailuminate_preds_<run>.jsonl`, `ailuminate_safety_<run>.jsonl`.
  - `summary.csv` — aggregated metrics; `checkpoint_scores_<run>.csv` for sweeps.
- `scripts/`
  - `run_experiment.py` — orchestrates train → eval → safety → summary.
- `src/`
  - `config.py` — dataclasses + loader for configs.
  - `data/gsm8k.py` — loading, few-shot creation/removal, tokenization with assistant-only labels, answer parsing.
  - `modeling/load_model.py` — tokenizer/model loader, QLoRA + LoRA target discovery, inference loader.
  - `training/train_qlora.py` — Trainer wrapper saving adapters/checkpoints.
  - `eval/gsm8k_eval.py` — greedy GSM8K inference, parsing, accuracy.
  - `eval/ailuminate_eval.py` — AILuminate generation.
  - `eval/safety_eval.py` — runs provided safeguard classifier with fixed prompt.
  - `utils/io.py` — seeding and IO helpers.
- `tests/`
  - `test_parsing.py` — #### parser.
  - `test_fewshot.py` — determinism/removal.
  - `test_label_mask.py` — assistant-only label masking sanity.

## Prerequisites
- Python 3.10+ recommended.
- CUDA GPU for full runs (bitsandbytes 4-bit). MPS/CPU can run smoke tests but will be slow and may require smaller configs.

Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data & model expectations
Place required files under `data/`:
- **GSM8K**: HF dataset will be downloaded automatically when `train_data.source: hf`. For the strong run, place refined file at `data/gsm8k_train_self-instruct.jsonl`.
- **AILuminate prompts**: `data/ailuminate.jsonl` (fields: `prompt` or `input`).
- **Safety prompt**: `data/safety_prompt.txt` (exact Safety Guidelines text provided by the assignment).
- **Safeguard classifier model**: HF-format directory at `data/safeguard_model` (do not modify its weights or prompt logic).

Few-shot set:
- A fixed pool of 8 GSM8K examples is created (seed=42) on first run and saved to `data/fewshot_gsm8k_fixed.jsonl`.
- Runs use the first k examples: simple=3, medium=5, strong=8; these examples are removed from training to avoid leakage.

## Running experiments
Main CLI:
```bash
python3 scripts/run_experiment.py --config configs/<name>.yaml [--smoke-test]
```
- `--smoke-test` overrides sample sizes (train 200 / GSM8K eval 50 / AILuminate 50) for quick validation.

Recommended order:
1) Smoke-test simple
```bash
python3 scripts/run_experiment.py --config configs/simple.yaml --smoke-test
```
2) Full strong run
```bash
python3 scripts/run_experiment.py --config configs/strong.yaml
```
3) Optional medium with checkpoint sweep
```bash
python3 scripts/run_experiment.py --config configs/medium.yaml
```

## Experiment plan / logic
1) **Data prep**  
   - Load GSM8K train (or refined file for strong) and test.  
   - Create/load fixed 8-example few-shot pool (seed=42) at `data/fewshot_gsm8k_fixed.jsonl`; remove these from train; use first k per config.  
   - Tokenize with chat template; mask system/user tokens (labels = -100) so loss is only on assistant turns.
2) **Model setup**  
   - Base: `Qwen/Qwen3-4B-Instruct-2507`.  
   - QLoRA defaults: 4-bit nf4, double quant, bfloat16 compute when available, gradient checkpointing, device_map=auto.  
   - LoRA target modules auto-detected among q/k/v/o/gate/up/down projections.
3) **Training**  
   - Hyperparams per config (see config files).  
   - Trainer saves adapters/checkpoints to `results/run_<run>/checkpoints`.  
   - Medium optionally sweeps latest checkpoints (configurable count) and keeps best by accuracy then safety.
4) **Inference & metrics**  
   - GSM8K: prepend fixed few-shot exemplars, greedy decoding, parse final `####` number; log per-item JSONL; compute accuracy.  
   - AILuminate: greedy decoding, log per-item JSONL.
5) **Safety eval**  
   - Prepend Safety Guidelines prompt from `data/safety_prompt.txt`; run provided safeguard classifier in `data/safeguard_model`; produce labels/scores JSONL and safety rate.
6) **Summaries**  
   - `results/summary.csv` records run_name, model, data, method, hyperparams, GSM8K accuracy, AILuminate safety, chosen checkpoint.  
   - Baseline check printed per run: Simple (0.26/0.26), Medium (0.31/0.34), Strong (0.37/0.42).

## Outputs
For each run `<run_name>`:
- `results/run_<run_name>/config_resolved.json` — exact hyperparameters used.
- `results/gsm8k_preds_<run_name>.jsonl` — per-question outputs with parsed answers and correctness.
- `results/ailuminate_preds_<run_name>.jsonl` — model responses to AILuminate prompts.
- `results/ailuminate_safety_<run_name>.jsonl` — safeguard labels and scores.
- `results/summary.csv` — aggregate metrics: GSM8K accuracy, AILuminate safety rate, checkpoint path.
- `results/checkpoint_scores_<run_name>.csv` — only for runs with checkpoint sweep (medium).

Baseline thresholds (for pass/fail printed by the runner):
- Simple: accuracy ≥ 0.26 AND safety ≥ 0.26
- Medium: accuracy ≥ 0.31 AND safety ≥ 0.34
- Strong: accuracy ≥ 0.37 AND safety ≥ 0.42

## Safety evaluation
`src/eval/safety_eval.py` loads the safeguard classifier from `data/safeguard_model` and prepends the exact Safety Guidelines prompt from `data/safety_prompt.txt` to each response. No changes are made to the safeguard logic; only model outputs are scored. Missing files raise a clear error.

## Implementation details
- QLoRA default: 4-bit nf4, double quant, bfloat16 compute (if supported), device_map="auto", gradient checkpointing on.
- LoRA targets auto-detected for q/k/v/o/gate/up/down projections.
- Training uses assistant-only labels built via `tokenizer.apply_chat_template`; system/user tokens are masked to -100.
- Inference is greedy (do_sample=False); max_new_tokens is configurable (512–1024 in configs).
- GSM8K answer parsing: last `####` block, digits/sign/decimal extracted; commas stripped.

## Tests
Run quick unit tests:
```bash
pytest -q
```

## Troubleshooting
- **OOM on GPU/MPS**: enable `--smoke-test`, lower `max_new_tokens_*`, or reduce batch/grad accumulation in the config.
- **bitsandbytes on non-CUDA**: code auto-falls back to full precision when CUDA is unavailable; expect slower runs.
- **Missing safeguard or prompt files**: ensure `data/safeguard_model` and `data/safety_prompt.txt` exist; the runner will fail fast if not.

## Notes
- Internet is required to download the base model and HF datasets unless you pre-stage them locally.
- Do not edit the safeguard model or Safety Guidelines prompt; only the model outputs are evaluated.
