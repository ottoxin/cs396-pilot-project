import torch
from transformers import AutoTokenizer

from src.data.gsm8k import _build_label_mask, SYSTEM_PROMPT


def test_label_mask_lengths():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "It is 2. #### 2"},
    ]
    input_ids, labels = _build_label_mask(tokenizer, messages, max_length=128)
    assert len(input_ids) == len(labels)
    # first tokens should be masked -100 until assistant start
    first_label = next((i for i, v in enumerate(labels) if v != -100), None)
    assert first_label is not None
    assert all(v == -100 for v in labels[:first_label])
    assert all(v != -100 for v in labels[first_label:])
