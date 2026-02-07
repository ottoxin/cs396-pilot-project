import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.data.gsm8k import parse_gsm8k_answer, build_messages, SYSTEM_PROMPT
from src.utils.io import ensure_dir


def evaluate_gsm8k(model, tokenizer, dataset, fewshot_examples: List[Dict], max_new_tokens: int, output_path: str) -> float:
    ensure_dir(os.path.dirname(output_path))
    preds = []
    correct = 0

    device = next(model.parameters()).device

    for idx, ex in enumerate(tqdm(dataset, desc="GSM8K eval")):
        messages = build_messages(ex["question"], fewshot_examples)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_text = tokenizer.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
        parsed_pred = parse_gsm8k_answer(output_text)
        gold = parse_gsm8k_answer(ex["answer"])
        is_correct = parsed_pred is not None and gold is not None and parsed_pred == gold
        correct += int(is_correct)
        preds.append(
            {
                "id": idx,
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "model_output": output_text,
                "parsed_final": parsed_pred,
                "correct": is_correct,
            }
        )

    acc = correct / len(dataset) if len(dataset) > 0 else 0.0
    with open(output_path, "w") as f:
        for rec in preds:
            f.write(json.dumps(rec) + "\n")
    return acc
