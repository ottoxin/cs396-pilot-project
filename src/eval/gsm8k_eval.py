import json
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.data.gsm8k import parse_gsm8k_answer, build_messages, SYSTEM_PROMPT
from src.utils.io import ensure_dir


def _batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset,
    fewshot_examples: List[Dict],
    max_new_tokens: int,
    output_path: str,
    batch_size: int = 1,
) -> float:
    ensure_dir(os.path.dirname(output_path))
    preds = []
    correct = 0

    device = next(model.parameters()).device
    batch_size = max(1, int(batch_size))
    dataset_list = list(dataset)

    row_idx = 0
    progress = tqdm(total=len(dataset_list), desc="GSM8K eval")
    for batch in _batched(dataset_list, batch_size):
        prompts = []
        for ex in batch:
            messages = build_messages(ex["question"], fewshot_examples)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = input_ids.shape[-1]

        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        continuations = gen[:, prompt_len:]
        outputs = tokenizer.batch_decode(continuations, skip_special_tokens=True)

        for ex, output_text in zip(batch, outputs):
            parsed_pred = parse_gsm8k_answer(output_text)
            gold = parse_gsm8k_answer(ex["answer"])
            is_correct = parsed_pred is not None and gold is not None and parsed_pred == gold
            correct += int(is_correct)
            preds.append(
                {
                    "id": row_idx,
                    "question": ex["question"],
                    "gold_answer": ex["answer"],
                    "model_output": output_text,
                    "parsed_final": parsed_pred,
                    "correct": is_correct,
                }
            )
            row_idx += 1
        progress.update(len(batch))
    progress.close()

    acc = correct / len(dataset) if len(dataset) > 0 else 0.0
    with open(output_path, "w") as f:
        for rec in preds:
            f.write(json.dumps(rec) + "\n")
    return acc
