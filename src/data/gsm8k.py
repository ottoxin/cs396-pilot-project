import json
import os
import random
from typing import Dict, List, Tuple

from datasets import load_dataset, Dataset

from src.utils.io import ensure_dir

SYSTEM_PROMPT = "You are a helpful assistant. Solve the math problem step by step. Give the final answer in the format '#### <number>'."
MAX_FEWSHOT = 8


def load_gsm8k(train_cfg, smoke_cfg) -> Tuple[Dataset, Dataset]:
    """Load GSM8K train (per config) and test splits."""
    if train_cfg.source == "hf":
        ds = load_dataset(train_cfg.path, "main")
        train_ds = ds[train_cfg.split]
    elif train_cfg.source == "file":
        train_ds = load_dataset("json", data_files=train_cfg.path)["train"]
    else:
        raise ValueError(f"Unknown train data source: {train_cfg.source}")

    test_ds = load_dataset("gsm8k", "main")["test"]

    if smoke_cfg.enabled:
        train_ds = train_ds.select(range(min(smoke_cfg.train_samples, len(train_ds))))
        test_ds = test_ds.select(range(min(smoke_cfg.gsm8k_eval_samples, len(test_ds))))

    return train_ds, test_ds


def prepare_fewshot(train_ds: Dataset, k: int, fewshot_path: str, seed: int) -> Tuple[Dataset, List[Dict]]:
    """Create or load fixed few-shot examples and remove them from training set.
    Always stores up to MAX_FEWSHOT examples; per-run uses first k."""
    ensure_dir(os.path.dirname(fewshot_path) or ".")

    if os.path.exists(fewshot_path):
        with open(fewshot_path, "r") as f:
            fewshot_examples = [json.loads(line) for line in f]
        if len(fewshot_examples) < k:
            raise ValueError(f"Fewshot file has {len(fewshot_examples)} examples, but k={k}")
    else:
        rng = random.Random(seed)
        need = min(MAX_FEWSHOT, len(train_ds))
        if need < k:
            raise ValueError("fewshot_k larger than training set")
        idxs = rng.sample(range(len(train_ds)), need)
        fewshot_examples = [train_ds[i] for i in idxs]
        with open(fewshot_path, "w") as f:
            for ex in fewshot_examples:
                f.write(json.dumps({"question": ex["question"], "answer": ex["answer"]}) + "\n")
        idxs = set(idxs)
        train_ds = train_ds.filter(lambda _, idx: idx not in idxs, with_indices=True)
        return train_ds, fewshot_examples[:k]

    # remove any overlapping training rows when fewshot file already exists
    few_qas = {(ex["question"], ex["answer"]) for ex in fewshot_examples}
    train_ds = train_ds.filter(lambda ex: (ex["question"], ex["answer"]) not in few_qas)
    return train_ds, fewshot_examples[:k]


def _build_label_mask(tokenizer, messages, max_length: int):
    # prompt without assistant content
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )[0]
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_tensors="pt"
    )[0]

    if full_ids.shape[0] > max_length:
        full_ids = full_ids[-max_length:]
    prompt_len = min(prompt_ids.shape[0], full_ids.shape[0])

    labels = full_ids.clone()
    labels[:prompt_len] = -100
    return full_ids.tolist(), labels.tolist()


def build_train_dataset(train_ds: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize GSM8K train set with assistant-only labels."""

    def map_example(ex):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        input_ids, labels = _build_label_mask(tokenizer, messages, max_length)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    tokenized = train_ds.map(map_example, remove_columns=train_ds.column_names)
    return tokenized


def parse_gsm8k_answer(text: str):
    """Parse final answer after the last ####. Returns str of digits or None."""
    if text is None:
        return None
    if "####" not in text:
        return None
    tail = text.split("####")[-1]
    # remove spaces and commas
    tail = tail.strip().replace(",", "")
    # keep leading sign and digits/decimal
    num = ""
    for ch in tail:
        if ch.isdigit() or ch in {"-", "."}:
            num += ch
        elif num:
            break
    return num if num else None


def build_messages(question: str, fewshot_examples: List[Dict]) -> List[Dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return messages
