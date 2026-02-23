#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.gsm8k import parse_gsm8k_answer


EXPECTED_TOTALS = {
    "simple": {"gsm8k_total": 1319, "ailuminate_total": 240},
    "medium": {"gsm8k_total": 263, "ailuminate_total": 240},
    "strong": {"gsm8k_total": 263, "ailuminate_total": 240},
}


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_gsm8k_metrics(path: str) -> Dict:
    rows = load_jsonl(path)
    total = len(rows)
    correct = 0
    for row in rows:
        if "correct" in row:
            correct += int(bool(row["correct"]))
        else:
            pred = parse_gsm8k_answer(row.get("model_output", ""))
            gold = parse_gsm8k_answer(row.get("gold_answer", ""))
            correct += int(pred is not None and gold is not None and pred == gold)
    accuracy = (correct / total) if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def compute_safety_metrics(path: str) -> Dict:
    rows = load_jsonl(path)
    total = len(rows)
    safe = 0
    unsafe = 0
    unknown = 0
    for row in rows:
        label = str(row.get("safety_label", "")).strip().lower()
        if label == "safe":
            safe += 1
        elif label == "unsafe":
            unsafe += 1
        else:
            unknown += 1
    safety_rate = (safe / total) if total else 0.0
    return {
        "safe": safe,
        "unsafe": unsafe,
        "unknown": unknown,
        "total": total,
        "safety_rate": safety_rate,
    }


def validate_totals(run_name: str, gsm8k_total: int, ailuminate_total: int) -> None:
    expected = EXPECTED_TOTALS[run_name]
    if gsm8k_total != expected["gsm8k_total"]:
        raise ValueError(
            f"{run_name} GSM8K total mismatch: got {gsm8k_total}, expected {expected['gsm8k_total']}"
        )
    if ailuminate_total != expected["ailuminate_total"]:
        raise ValueError(
            f"{run_name} AILuminate total mismatch: got {ailuminate_total}, expected {expected['ailuminate_total']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate submission metrics for simple/medium/strong runs.")
    parser.add_argument("--output-dir", default="submission_results", help="Run output root directory.")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    final_rows: List[Dict] = []
    for run_name in ("simple", "medium", "strong"):
        run_dir = os.path.join(output_dir, f"run_{run_name}")
        gsm8k_path = os.path.join(run_dir, f"gsm8k_preds_{run_name}.jsonl")
        safety_path = os.path.join(run_dir, f"ailuminate_safety_{run_name}.jsonl")

        if not os.path.exists(gsm8k_path):
            raise FileNotFoundError(f"Missing GSM8K predictions: {gsm8k_path}")
        if not os.path.exists(safety_path):
            raise FileNotFoundError(f"Missing AILuminate safety results: {safety_path}")

        gsm8k = compute_gsm8k_metrics(gsm8k_path)
        safety = compute_safety_metrics(safety_path)
        validate_totals(run_name, gsm8k["total"], safety["total"])

        final_rows.append(
            {
                "run_name": run_name,
                "gsm8k_correct": gsm8k["correct"],
                "gsm8k_total": gsm8k["total"],
                "gsm8k_accuracy": gsm8k["accuracy"],
                "ailuminate_safe": safety["safe"],
                "ailuminate_unsafe": safety["unsafe"],
                "ailuminate_unknown": safety["unknown"],
                "ailuminate_total": safety["total"],
                "ailuminate_safety_rate": safety["safety_rate"],
            }
        )

    json_path = os.path.join(output_dir, "final_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)

    csv_path = os.path.join(output_dir, "final_metrics.csv")
    fieldnames = [
        "run_name",
        "gsm8k_correct",
        "gsm8k_total",
        "gsm8k_accuracy",
        "ailuminate_safe",
        "ailuminate_unsafe",
        "ailuminate_unknown",
        "ailuminate_total",
        "ailuminate_safety_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
