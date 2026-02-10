import json
import os

import torch
from tqdm import tqdm

from src.utils.io import ensure_dir


def _extract_prompt(example: dict) -> str:
    for key in ("prompt", "input", "prompt_text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("AILuminate example missing prompt field (expected one of: prompt, input, prompt_text)")


def run_ailuminate(model, tokenizer, dataset, max_new_tokens: int, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    model.eval()
    device = getattr(model, "device", None) or next(model.parameters()).device

    with open(output_path, "w") as f:
        for idx, ex in enumerate(tqdm(dataset, desc="AILuminate eval")):
            prompt = _extract_prompt(ex)
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
            rec = {"id": idx, "prompt": prompt, "response": output_text}
            f.write(json.dumps(rec) + "\n")
            f.flush()
