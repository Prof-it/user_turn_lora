"""
UserTurnLoRA - Fine-tuning LLMs to predict user turns in conversations.
"""

from .config import PipelineConfig, get_config, SPECIAL_TOKENS, SYSTEM_PROMPT
from .data import load_data, format_for_training, build_eval_examples
from .model import (
    load_tokenizer,
    load_base_model,
    load_finetuned_model,
    predict_next_user,
    cleanup_model,
)
from .train import train, train_from_pairs
from .evaluate import (
    evaluate_baseline,
    evaluate_finetuned,
    compare_results,
)

__all__ = [
    "PipelineConfig",
    "get_config",
    "SPECIAL_TOKENS",
    "SYSTEM_PROMPT",
    "load_data",
    "format_for_training",
    "build_eval_examples",
    "load_tokenizer",
    "load_base_model",
    "load_finetuned_model",
    "predict_next_user",
    "cleanup_model",
    "train",
    "train_from_pairs",
    "evaluate_baseline",
    "evaluate_finetuned",
    "compare_results",
]
