#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m src.data.prepare_ailuminate_data

python -m src.cli.run_experiment \
  --config configs/simple.yaml \
  --checkpoint haohxin/simple_pilot_project_model \
  --full-gsm8k-test \
  --rerun-all \
  --output-dir submission_results

python -m src.cli.run_experiment \
  --config configs/medium.yaml \
  --checkpoint haohxin/medium_pilot_project_model \
  --rerun-all \
  --output-dir submission_results

python -m src.cli.run_experiment \
  --config configs/strong.yaml \
  --checkpoint haohxin/strong_pilot_project_model \
  --rerun-all \
  --output-dir submission_results

python scripts/aggregate_submission_metrics.py --output-dir submission_results

echo "Submission metrics written to:"
echo "  submission_results/final_metrics.json"
echo "  submission_results/final_metrics.csv"
