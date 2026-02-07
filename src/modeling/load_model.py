import os
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


TARGET_MODULE_KEYWORDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _detect_target_modules(model) -> List[str]:
    targets = set()
    for name, module in model.named_modules():
        if any(key in name for key in TARGET_MODULE_KEYWORDS):
            # only pick linear layers
            if hasattr(module, "weight"):
                targets.add(name.split(".")[-1])
    # fallback
    if not targets:
        targets = {"q_proj", "v_proj"}
    return sorted(targets)


def _compute_dtype(pref: str):
    if pref == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if pref == "bfloat16" and torch.backends.mps.is_available():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(cfg, for_training: bool = True):
    dtype = _compute_dtype(cfg.quantization.compute_dtype)

    bnb_config = None
    use_bnb = bool(getattr(cfg.quantization, "use_bnb", True)) and torch.cuda.is_available()
    if use_bnb:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=cfg.quantization.double_quant,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4" if cfg.quantization.nf4 else "fp4",
            )
        except Exception:
            bnb_config = None
            use_bnb = False

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=bnb_config,
    )

    if for_training and use_bnb:
        model = prepare_model_for_kbit_training(model)
    if for_training:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    targets = _detect_target_modules(model)
    peft_config = LoraConfig(
        r=cfg.train_hparams.lora_r,
        lora_alpha=cfg.train_hparams.lora_alpha,
        lora_dropout=cfg.train_hparams.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_for_inference(base_model: str, adapter_path: Optional[str], quant_config, device_map="auto"):
    dtype = _compute_dtype(quant_config.compute_dtype)
    bnb_config = None
    use_bnb = getattr(quant_config, "use_bnb", True) and torch.cuda.is_available()
    if use_bnb:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=quant_config.double_quant,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4" if quant_config.nf4 else "fp4",
            )
        except Exception:
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model
