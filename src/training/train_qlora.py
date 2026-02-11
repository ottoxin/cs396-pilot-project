import os
from typing import Optional, Tuple

from transformers import Trainer, TrainingArguments, default_data_collator, DataCollatorForSeq2Seq

from src.utils.io import ensure_dir

def train_model(
    cfg,
    model,
    tokenizer,
    train_dataset,
    run_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(checkpoint_dir)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=cfg.train_hparams.num_epochs,
        per_device_train_batch_size=cfg.train_hparams.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train_hparams.gradient_accumulation_steps,
        learning_rate=cfg.train_hparams.learning_rate,
        weight_decay=cfg.train_hparams.weight_decay,
        warmup_ratio=cfg.train_hparams.warmup_ratio,
        logging_steps=50,
        save_steps=cfg.checkpointing.save_steps,
        save_total_limit=cfg.checkpointing.sweep_max_checkpoints,
        eval_strategy="no", 
        save_strategy="steps",
        lr_scheduler_type="cosine",
        bf16=cfg.quantization.compute_dtype == "bfloat16",
        fp16=cfg.quantization.compute_dtype == "float16",
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final adapter and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(run_dir)
    return checkpoint_dir
