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


def _batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_ailuminate(model, tokenizer, dataset, max_new_tokens: int, output_path: str, batch_size: int = 1) -> None:
    ensure_dir(os.path.dirname(output_path))
    model.eval()
    device = getattr(model, "device", None) or next(model.parameters()).device
    batch_size = max(1, int(batch_size))
    dataset_list = list(dataset)

    with open(output_path, "w") as f:
        idx = 0
        progress = tqdm(total=len(dataset_list), desc="AILuminate eval")
        for batch in _batched(dataset_list, batch_size):
            prompts = [_extract_prompt(ex) for ex in batch]
            chat_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_prompts.append(chat_prompt)

            enc = tokenizer(
                chat_prompts,
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
            for prompt, output_text in zip(prompts, outputs):
                rec = {"id": idx, "prompt": prompt, "response": output_text}
                f.write(json.dumps(rec) + "\n")
                idx += 1
            progress.update(len(batch))
        progress.close()
