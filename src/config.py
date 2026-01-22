"""
Configuration module for UserTurnLoRA pipeline.
Centralizes all hyperparameters and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


SPECIAL_TOKENS = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "user_start": "<|im_start|>user",
        "assistant_start": "<|im_start|>assistant",
        "system_start": "<|im_start|>system",
        "end": "<|im_end|>"
    },
    "LiquidAI/LFM2.5-1.2B-Instruct": {
        "user_start": "<|im_start|>user",
        "assistant_start": "<|im_start|>assistant",
        "system_start": "<|im_start|>system",
        "end_of_text": "<|endoftext|>",
        "start_of_text": "<|startoftext|>",
        "end": "<|im_end|>"
    },
    "mistralai/Ministral-3-3B-Instruct-2512": {
        "user_start": "[INST]",
        "user_end": "[/INST]",
        "assistant_start": "",  # Mistral has no explicit assistant start token
        "system_start": "[SYSTEM_PROMPT]",
        "system_end": "[/SYSTEM_PROMPT]",
        "end": "</s>"
    },
    "allenai/OLMo-3-7B-Instruct": {
       "user_start": "<|start_header_id|>user<|end_header_id|>",
       "assistant_start": "<|start_header_id|>assistant<|end_header_id|>",
       "system_start": "<|start_header_id|>system<|end_header_id|>",
       "end": "<|eot_id|>"
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
       "user_start": "<|start_header_id|>user<|end_header_id|>",
       "assistant_start": "<|start_header_id|>assistant<|end_header_id|>",
       "system_start": "<|start_header_id|>system<|end_header_id|>",
       "end": "<|eot_id|>"
    },
}


# System prompt used for all models
SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class PipelineConfig:
    """Main configuration for the UserTurnLoRA pipeline."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # Data settings
    num_train_samples: int = 6000
    num_eval_samples: int = 400  # ~6.25% of total, used for human evaluation
    wildchat_ratio: float = 0.5  # 50% WildChat, 50% SGD
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Training settings (optimal from ablation study)
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    
    # LoRA settings (optimal from ablation study)
    lora_r: int = 8
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Paths
    output_dir: str = "output"
    adapter_subdir: str = "adapter"
    
    # Evaluation
    # max_context_len: Maximum tokens for input context (conversation history + target).
    # NOT the same as max_new_tokens (which controls generation length).
    # This truncates long conversations to fit within model context window.
    max_context_len: int = 4096
    
    # Hardware
    use_4bit: bool = True
    
    # Logging
    logging_steps: int = 1
    eval_steps: int = 10
    save_strategy: str = "epoch"
    report_to: str = "wandb"
    
    def __post_init__(self):
        """Validate and set derived attributes."""
        if self.model_name not in SPECIAL_TOKENS:
            raise ValueError(f"Unknown model: {self.model_name}. "
                           f"Supported: {list(SPECIAL_TOKENS.keys())}")
        
        # Auto-detect compute dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float16
        
        # Adjust batch size for fp16
        if self.compute_dtype == torch.float16:
            self.batch_size = 2
            self.gradient_accumulation_steps = 32
            self.lora_r = 16
            self.lora_alpha = 32
    
    @property
    def special_tokens(self) -> dict:
        """Get special tokens for the configured model."""
        tokens = SPECIAL_TOKENS[self.model_name].copy()
        # Remove **** workaround for actual use
        for key, value in tokens.items():
            if "****" in value:
                tokens[key] = value.replace("****", "")
        return tokens
    
    @property
    def adapter_path(self) -> str:
        """Full path to adapter directory."""
        return os.path.join(self.output_dir, self.adapter_subdir)
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            "model_name": self.model_name,
            "num_train_samples": self.num_train_samples,
            "num_eval_samples": self.num_eval_samples,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "compute_dtype": str(self.compute_dtype),
        }


def get_config(model_name: Optional[str] = None, **overrides) -> PipelineConfig:
    """
    Create a pipeline configuration.
    
    Args:
        model_name: HuggingFace model name (uses default if None)
        **overrides: Override any config parameter
    
    Returns:
        PipelineConfig instance
    """
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)
