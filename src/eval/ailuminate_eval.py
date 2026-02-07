import json
import os
from typing import List

import torch
from tqdm import tqdm

from src.utils.io import ensure_dir


def run_ailuminate(model, tokenizer, dataset, max_new_tokens: int, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    preds = []
    device = next(model.parameters()).device

    for idx, ex in enumerate(tqdm(dataset, desc="AILuminate eval")):
        prompt = ex["prompt"] if "prompt" in ex else ex.get("input", "")
        messages = [{"role": "user", "content": prompt}]
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
        preds.append({"id": idx, "prompt": prompt, "response": output_text})

    with open(output_path, "w") as f:
        for rec in preds:
            f.write(json.dumps(rec) + "\n")

