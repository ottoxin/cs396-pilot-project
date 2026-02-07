import json
import os
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.io import ensure_dir


def _load_safety_components(model_path: str, device: str = "auto"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        truncation=True,
    )
    return pipe


def run_safety_eval(safeguard_model_path: str, safety_prompt_path: str, preds_path: str, output_path: str) -> float:
    """
    Run the provided safeguard classifier on model generations.
    Assumes `safeguard_model_path` is a HF-compatible text-classification model directory
    and `safety_prompt_path` is the exact Safety Guidelines prompt to prepend.
    """
    ensure_dir(os.path.dirname(output_path))

    if not os.path.exists(safeguard_model_path):
        raise FileNotFoundError(f"safeguard model path not found: {safeguard_model_path}")
    if not os.path.exists(safety_prompt_path):
        raise FileNotFoundError(f"safety prompt not found: {safety_prompt_path}")

    with open(safety_prompt_path, "r") as f:
        safety_prompt = f.read().strip()

    with open(preds_path, "r") as f:
        rows = [json.loads(line) for line in f]

    clf = _load_safety_components(safeguard_model_path)

    labeled: List[dict] = []
    safe = 0
    for row in rows:
        text = f"{safety_prompt}\n\nAssistant response:\n{row['response']}"
        result = clf(text, truncation=True, max_length=1024)[0]
        label = result.get("label", "").lower()
        is_safe = "safe" in label
        safe += int(is_safe)
        labeled.append({**row, "safety_label": label, "safety_score": result.get("score", None)})

    with open(output_path, "w") as f:
        for rec in labeled:
            f.write(json.dumps(rec) + "\n")

    safety_rate = safe / len(rows) if rows else 0.0
    return safety_rate

