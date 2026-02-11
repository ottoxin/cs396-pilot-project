import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class TrainDataConfig:
    source: str  # 'hf' or 'file'
    path: str
    split: str = "train"


@dataclass
class TrainHparams:
    num_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float


@dataclass
class QuantConfig:
    use_bnb: bool = True
    double_quant: bool = True
    nf4: bool = True
    compute_dtype: str = "bfloat16"  # or float16


@dataclass
class CheckpointingConfig:
    save_steps: int = 500
    eval_steps: int = 500
    checkpoint_sweep: bool = False
    sweep_max_checkpoints: int = 3


@dataclass
class SmokeTestConfig:
    enabled: bool = False
    train_samples: int = 200
    gsm8k_eval_samples: int = 50
    ailuminate_eval_samples: int = 50


@dataclass
class DataPaths:
    fewshot_file: str
    refined_gsm8k: str
    ailuminate: str
    safety_prompt: str
    safeguard_model: str


@dataclass
class ExperimentConfig:
    run_name: str
    base_model: str
    seed: int = 42
    train_data: TrainDataConfig = field(default_factory=TrainDataConfig)
    method: str = "qlora"
    train_hparams: TrainHparams = field(default_factory=TrainHparams)
    max_seq_length: int = 2048
    fewshot_k: int = 3
    max_new_tokens_gsm8k: int = 512
    max_new_tokens_ailuminate: int = 512
    eval_batch_size: int = 1
    quantization: QuantConfig = field(default_factory=QuantConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)
    data_paths: DataPaths = field(default_factory=DataPaths)
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_resolved(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    return cls(**data)


def load_config(path: str, smoke_override: Optional[bool] = None) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Build nested configs
    train_data = _dict_to_dataclass(TrainDataConfig, raw["train_data"])
    train_hparams = _dict_to_dataclass(TrainHparams, raw["train_hparams"])
    quantization = _dict_to_dataclass(QuantConfig, raw.get("quantization", {}))
    checkpointing = _dict_to_dataclass(CheckpointingConfig, raw.get("checkpointing", {}))
    smoke_test = _dict_to_dataclass(SmokeTestConfig, raw.get("smoke_test", {}))
    data_paths = _dict_to_dataclass(DataPaths, raw.get("data_paths", {}))

    if smoke_override is not None:
        smoke_test.enabled = smoke_override

    cfg = ExperimentConfig(
        run_name=raw["run_name"],
        base_model=raw["base_model"],
        seed=raw.get("seed", 42),
        train_data=train_data,
        method=raw.get("method", "qlora"),
        train_hparams=train_hparams,
        max_seq_length=raw.get("max_seq_length", 2048),
        fewshot_k=raw.get("fewshot_k", 3),
        max_new_tokens_gsm8k=raw.get("max_new_tokens_gsm8k", 512),
        max_new_tokens_ailuminate=raw.get("max_new_tokens_ailuminate", 512),
        eval_batch_size=raw.get("eval_batch_size", 1),
        quantization=quantization,
        checkpointing=checkpointing,
        smoke_test=smoke_test,
        data_paths=data_paths,
        output_dir=raw.get("output_dir", "results"),
    )
    return cfg


__all__ = [
    "ExperimentConfig",
    "TrainDataConfig",
    "TrainHparams",
    "QuantConfig",
    "CheckpointingConfig",
    "SmokeTestConfig",
    "DataPaths",
    "load_config",
]
