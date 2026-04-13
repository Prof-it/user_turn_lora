"""
Model runners for multi-turn rollout evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from .config import PipelineConfig, SYSTEM_PROMPT, get_config
from .config_loader import load_saved_pipeline_config
from .generation import generate_from_messages
from .model import (
    cleanup_model,
    load_base_model,
    load_finetuned_model,
    load_tokenizer,
    predict_next_user,
)
from .prompting import build_prompt_messages, load_few_shot_examples


class RuntimeHandle:
    """Simple runtime wrapper with a callable generator and cleanup hook."""

    def __init__(self, *, generate: Callable[[list], str], cleanup: Callable[[], None], metadata: Dict):
        self.generate = generate
        self.cleanup = cleanup
        self.metadata = metadata


def _blocked_role_texts(config: PipelineConfig) -> list[str]:
    tokens = config.special_tokens
    return [tokens.get("assistant_start", ""), tokens.get("system_start", ""), tokens.get("user_start", "")]


def load_user_runtime(
    model_dir: Path,
    *,
    simulator_kind: str,
    num_few_shot_examples: int = 3,
) -> RuntimeHandle:
    """Load a user simulator from an existing model directory."""
    config = load_saved_pipeline_config(model_dir)
    few_shot_examples = None

    if simulator_kind == "finetuned":
        model, tokenizer = load_finetuned_model(config, adapter_path=str(model_dir / "adapter"))

        def generate(history: list) -> str:
            return predict_next_user(model, tokenizer, {"conversation": history}, config)

    elif simulator_kind in {"base", "prompt_zero_shot", "prompt_few_shot"}:
        tokenizer = load_tokenizer(config)
        model = load_base_model(config, for_training=False)
        blocked_texts = [text for text in _blocked_role_texts(config) if text]

        if simulator_kind == "base":
            def generate(history: list) -> str:
                return predict_next_user(model, tokenizer, {"conversation": history}, config)

        else:
            mode = "few-shot" if simulator_kind == "prompt_few_shot" else "zero-shot"
            if mode == "few-shot":
                few_shot_examples = load_few_shot_examples(
                    num_examples=num_few_shot_examples,
                    model_dir=model_dir,
                )

            def generate(history: list) -> str:
                messages = build_prompt_messages(history, mode, few_shot_examples)
                return generate_from_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    config=config,
                    blocked_texts=blocked_texts,
                )

    else:
        raise ValueError(f"Unknown simulator_kind: {simulator_kind}")

    return RuntimeHandle(
        generate=generate,
        cleanup=lambda: cleanup_model(model=model),
        metadata={
            "model_name": config.model_name,
            "simulator_kind": simulator_kind,
            "num_few_shot_examples": num_few_shot_examples if few_shot_examples else 0,
        },
    )


def load_assistant_runtime(
    *,
    model_name: str,
    temperature: float = 0.2,
    max_new_tokens: int = 128,
    do_sample: bool = True,
) -> RuntimeHandle:
    """Load a fixed assistant model for free closed-loop rollout."""
    config = get_config(
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=False)
    blocked_texts = [config.special_tokens.get("user_start", ""), config.special_tokens.get("system_start", "")]
    blocked_texts = [text for text in blocked_texts if text]

    def generate(history: list) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        return generate_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            config=config,
            blocked_texts=blocked_texts,
        )

    return RuntimeHandle(
        generate=generate,
        cleanup=lambda: cleanup_model(model=model),
        metadata={
            "assistant_model_name": model_name,
            "assistant_temperature": temperature,
            "assistant_max_new_tokens": max_new_tokens,
        },
    )
