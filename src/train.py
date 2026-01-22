"""
Training module for UserTurnLoRA pipeline.
Handles SFT training with LoRA adapters.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .config import PipelineConfig
from .model import load_base_model, load_tokenizer, get_lora_config, cleanup_model

# W&B project name
WANDB_PROJECT = "userturn-lora"


def get_training_config(config: PipelineConfig) -> SFTConfig:
    """Create SFT training configuration."""
    use_bf16 = config.compute_dtype == torch.bfloat16
    
    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        packing=False,
        gradient_checkpointing=True,
        group_by_length=False,
        label_smoothing_factor=0.0,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=config.report_to,
    )


def train(
    config: PipelineConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """
    Train a LoRA adapter on the base model.
    
    Args:
        config: Pipeline configuration
        train_dataset: Training dataset with input_ids and labels
        eval_dataset: Evaluation dataset
        resume_from_checkpoint: Path to checkpoint to resume from
    
    Returns:
        Path to saved adapter
    """
    # Initialize wandb if enabled
    if config.report_to == "wandb":
        import wandb
        model_short = config.model_name.split("/")[-1]
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{model_short}-{config.num_train_samples}samples",
            config=config.to_dict(),
            reinit=True,
        )
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print("="*60 + "\n")
    
    # Load model and tokenizer
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=True)
    
    # Set model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Get configs
    peft_config = get_lora_config(config)
    train_config = get_training_config(config)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # Train
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save adapter
        adapter_path = config.adapter_path
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        print(f"\nAdapter saved to {adapter_path}")
        
    finally:
        # Cleanup
        cleanup_model(model=model, trainer=trainer)
    
    return adapter_path


def train_from_pairs(
    config: PipelineConfig,
    train_pairs: list,
    eval_pairs: list,
) -> str:
    """
    Train from raw conversation pairs.
    
    Convenience function that handles data formatting.
    
    Args:
        config: Pipeline configuration
        train_pairs: List of training conversation pairs
        eval_pairs: List of evaluation conversation pairs
    
    Returns:
        Path to saved adapter
    """
    from .data import format_for_training
    
    tokenizer = load_tokenizer(config)
    
    print("Formatting training data...")
    train_ds = format_for_training(train_pairs, tokenizer, config)
    eval_ds = format_for_training(eval_pairs, tokenizer, config)
    
    return train(config, train_ds, eval_ds)
