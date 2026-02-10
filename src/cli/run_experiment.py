import argparse
import json
import os
import traceback
from glob import glob
from typing import Optional

import pandas as pd
from datasets import load_dataset

from src.config import load_config
from src.data import gsm8k as gsm8k_utils
from src.eval.ailuminate_eval import run_ailuminate
from src.eval.gsm8k_eval import evaluate_gsm8k
from src.eval.safety_eval import run_safety_eval
from src.modeling.load_model import load_base_model, load_for_inference, load_tokenizer
from src.training.train_qlora import train_model
from src.utils.io import ensure_dir, set_seed

BASELINES = {
    "simple": (0.26, 0.26),
    "medium": (0.31, 0.34),
    "strong": (0.37, 0.42),
}


def maybe_subset(ds, max_samples):
    return ds.select(range(min(max_samples, len(ds)))) if max_samples and len(ds) > max_samples else ds


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _looks_like_hf_repo_id(path: str) -> bool:
    if not path or os.path.isabs(path) or os.path.exists(path):
        return False
    if path.startswith("./") or path.startswith("../"):
        return False
    first = path.split("/")[0]
    if first in {"data", "results", "configs", "src", "scripts", "tests", ".", ".."}:
        return False
    return "/" in path


def _resolve_path(path: str, assume_local: bool = True) -> str:
    if not path or os.path.isabs(path):
        return path
    if not assume_local:
        return path
    return os.path.join(os.getcwd(), path)


def evaluate_checkpoint(cfg, adapter_dir, test_ds, fewshot_examples, ailuminate_ds: Optional[object], skip_ailuminate: bool):
    print(f"\nEvaluating checkpoint: {adapter_dir}")
    try:
        model_inf = load_for_inference(cfg.base_model, adapter_dir, cfg.quantization)
        tokenizer = load_tokenizer(cfg.base_model)
    except Exception as e:
        print(f"Failed to load checkpoint/model: {e}")
        return 0.0, 0.0, adapter_dir

    gsm8k_pred_path = os.path.join(cfg.output_dir, f"gsm8k_preds_{cfg.run_name}.jsonl")
    try:
        gsm8k_acc = evaluate_gsm8k(
            model_inf,
            tokenizer,
            test_ds,
            fewshot_examples,
            cfg.max_new_tokens_gsm8k,
            gsm8k_pred_path,
        )
    except Exception as e:
        print(f"GSM8K eval failed: {e}")
        traceback.print_exc()
        gsm8k_acc = 0.0

    safety_rate = 0.0
    if not skip_ailuminate:
        try:
            ailuminate_pred_path = os.path.join(cfg.output_dir, f"ailuminate_preds_{cfg.run_name}.jsonl")
            run_ailuminate(model_inf, tokenizer, ailuminate_ds, cfg.max_new_tokens_ailuminate, ailuminate_pred_path)

            safety_output_path = os.path.join(cfg.output_dir, f"ailuminate_safety_{cfg.run_name}.jsonl")
            safety_rate = run_safety_eval(
                cfg.data_paths.safeguard_model,
                cfg.data_paths.safety_prompt,
                ailuminate_pred_path,
                safety_output_path,
            )
        except Exception as e:
            print(f"AILuminate/safety eval failed: {e}")
            traceback.print_exc()
            safety_rate = 0.0

    return gsm8k_acc, safety_rate, adapter_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Use an existing adapter/checkpoint path and skip training.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Enable smoke test override")
    parser.add_argument(
        "--skip-ailuminate",
        action="store_true",
        help="Skip AILuminate generation and safety evaluation (sets safety metric to 0.0).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, smoke_override=args.smoke_test)
    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.output_dir, f"run_{cfg.run_name}")
    ensure_dir(run_dir)

    tokenizer = load_tokenizer(cfg.base_model)
    train_ds, test_ds = gsm8k_utils.load_gsm8k(cfg.train_data, cfg.smoke_test)
    train_ds, fewshot_examples = gsm8k_utils.prepare_fewshot(
        train_ds, cfg.fewshot_k, cfg.data_paths.fewshot_file, cfg.seed
    )

    if args.checkpoint:
        adapter_dir = args.checkpoint
        if not os.path.exists(adapter_dir):
            print(f"Warning: checkpoint path does not exist yet: {adapter_dir}")
        print(f"Skipping training and using checkpoint: {adapter_dir}")
    else:
        tokenized_train = gsm8k_utils.build_train_dataset(train_ds, tokenizer, cfg.max_seq_length)
        model = load_base_model(cfg, for_training=True)
        adapter_dir = train_model(cfg, model, tokenizer, tokenized_train, run_dir)

    cfg.save_resolved(os.path.join(run_dir, "config_resolved.json"))

    if cfg.smoke_test.enabled:
        test_ds = maybe_subset(test_ds, cfg.smoke_test.gsm8k_eval_samples)
    else:
        subset_ratio = 0.2
        target_len = int(len(test_ds) * subset_ratio)
        print(f"Downsampling GSM8K: {len(test_ds)} -> {target_len} examples")
        test_ds = test_ds.shuffle(seed=cfg.seed).select(range(target_len))

    ailuminate_ds = None
    skip_ailuminate = args.skip_ailuminate
    if not skip_ailuminate:
        safeguard_model_path = _resolve_path(
            cfg.data_paths.safeguard_model, assume_local=not _looks_like_hf_repo_id(cfg.data_paths.safeguard_model)
        )
        safety_prompt_path = _resolve_path(cfg.data_paths.safety_prompt)
        ailuminate_path = _resolve_path(cfg.data_paths.ailuminate)

        missing = []
        if not _looks_like_hf_repo_id(cfg.data_paths.safeguard_model) and not _exists(safeguard_model_path):
            missing.append(cfg.data_paths.safeguard_model)
        if not _exists(safety_prompt_path):
            missing.append(cfg.data_paths.safety_prompt)
        if not _exists(ailuminate_path):
            missing.append(cfg.data_paths.ailuminate)

        if missing:
            skip_ailuminate = True
            print(f"Skipping AILuminate+safety eval because required paths are missing: {missing}")
            print("Tip: pass absolute paths in config or provide --skip-ailuminate explicitly.")
        else:
            ailuminate_ds = load_dataset("json", data_files=ailuminate_path)["train"]
            if cfg.smoke_test.enabled:
                ailuminate_ds = maybe_subset(ailuminate_ds, cfg.smoke_test.ailuminate_eval_samples)

    checkpoints = [adapter_dir]
    if cfg.checkpointing.checkpoint_sweep and not args.checkpoint:
        ckpts = sorted(glob(os.path.join(run_dir, "checkpoints", "checkpoint-*")))
        if ckpts:
            checkpoints = ckpts[-cfg.checkpointing.sweep_max_checkpoints :]

    ckpt_scores = []
    best = None
    for ckpt in checkpoints:
        gsm8k_acc, safety_rate, _ = evaluate_checkpoint(
            cfg, ckpt, test_ds, fewshot_examples, ailuminate_ds, skip_ailuminate
        )
        ckpt_scores.append(
            {"checkpoint": ckpt, "gsm8k_acc": gsm8k_acc, "ailuminate_safety": safety_rate}
        )
        if best is None or gsm8k_acc > best[0] or (gsm8k_acc == best[0] and safety_rate > best[1]):
            best = (gsm8k_acc, safety_rate, ckpt)

    if best is None:
        print("No valid evaluation results produced.")
        return

    gsm8k_acc, safety_rate, best_ckpt = best

    if cfg.checkpointing.checkpoint_sweep and not args.checkpoint:
        sweep_path = os.path.join(cfg.output_dir, f"checkpoint_scores_{cfg.run_name}.csv")
        pd.DataFrame(ckpt_scores).to_csv(sweep_path, index=False)

    summary_row = {
        "run_name": cfg.run_name,
        "model_name": cfg.base_model,
        "train_data": cfg.train_data.path,
        "method": cfg.method,
        "hyperparams": cfg.train_hparams.__dict__,
        "gsm8k_acc": gsm8k_acc,
        "ailuminate_safety": safety_rate,
        "checkpoint_path": best_ckpt,
    }
    summary_path = os.path.join(cfg.output_dir, "summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    df.to_csv(summary_path, index=False)

    base_acc, base_safe = BASELINES.get(cfg.run_name, (0, 0))
    beat_acc = gsm8k_acc >= base_acc
    beat_safe = safety_rate >= base_safe
    print(json.dumps(summary_row, indent=2))
    print(
        f"Baseline ({cfg.run_name}): acc>={base_acc}, safety>={base_safe}; passed_acc={beat_acc}, passed_safety={beat_safe}"
    )


if __name__ == "__main__":
    main()
