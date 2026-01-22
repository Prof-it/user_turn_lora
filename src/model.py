"""
Model loading and management module for UserTurnLoRA pipeline.
Handles base model loading, quantization, and LoRA adapter management.
"""

import gc
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
)
from transformers.generation.logits_process import NoBadWordsLogitsProcessor
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

from .config import PipelineConfig, SYSTEM_PROMPT


def get_quantization_config(config: PipelineConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config for 4-bit loading."""
    if not config.use_4bit:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(config: PipelineConfig) -> AutoTokenizer:
    """Load and configure tokenizer."""
    print(f"Loading tokenizer for {config.model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_base_model(
    config: PipelineConfig,
    for_training: bool = False
) -> AutoModelForCausalLM:
    """
    Load base model with quantization.
    
    Args:
        config: Pipeline configuration
        for_training: If True, prepare model for k-bit training
    
    Returns:
        Loaded model
    """
    print(f"Loading model {config.model_name}...")
    print(f"  Compute dtype: {config.compute_dtype}")
    print(f"  4-bit quantization: {config.use_4bit}")
    
    bnb_config = get_quantization_config(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=config.compute_dtype,
        trust_remote_code=True,
    )
    
    if for_training:
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model.eval()
        model.config.use_cache = True
    
    print(f"  Model loaded successfully")
    return model


def get_lora_config(config: PipelineConfig) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )


def load_finetuned_model(
    config: PipelineConfig,
    adapter_path: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model with fine-tuned LoRA adapter.
    
    Args:
        config: Pipeline configuration
        adapter_path: Path to adapter (uses config.adapter_path if None)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    adapter_path = adapter_path or config.adapter_path
    print(f"Loading fine-tuned model from {adapter_path}...")
    
    tokenizer = load_tokenizer(config)
    
    bnb_config = get_quantization_config(config)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=config.compute_dtype,
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.config.use_cache = True
    
    print(f"  Fine-tuned model loaded successfully")
    return model, tokenizer


def build_messages(item: dict, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build message list for chat template."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in item["conversation"]:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": m.get("content", "")})
    return msgs


@torch.no_grad()
def predict_next_user(
    model,
    tokenizer,
    item: dict,
    config: PipelineConfig,
    verbose: bool = False
) -> str:
    """
    Generate prediction for the next user turn.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        item: Dict with 'conversation' key
        config: Pipeline configuration
        verbose: Print debug info
    
    Returns:
        Predicted user turn text
    """
    messages = build_messages(item)
    
    # Tokenize conversation
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(model.device)
    
    # Append user start tokens
    special_tokens = config.special_tokens
    user_open_tokens = tokenizer.encode(
        special_tokens["user_start"] + "\n",
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    input_ids = torch.cat([inputs, user_open_tokens], dim=-1)
    attention_mask = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]
    
    # Prevent role headers from appearing in generation
    bad_words = tokenizer(
        [special_tokens["assistant_start"], 
         special_tokens["system_start"], 
         special_tokens["user_start"]],
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"].tolist()
    
    logits_processors = LogitsProcessorList([
        NoBadWordsLogitsProcessor(bad_words, eos_token_id=tokenizer.eos_token_id)
    ])
    
    # Generate
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        no_repeat_ngram_size=4,
        logits_processor=logits_processors,
    )
    
    gen_ids = outputs[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    if verbose:
        print(f"Generated: {text[:200]}...")
    
    return text


def cleanup_model(model=None, trainer=None, extra_vars=None):
    """
    Free GPU memory by cleaning up model and related objects.
    
    Args:
        model: Model to delete
        trainer: Trainer to cleanup
        extra_vars: List of additional variables to delete
    """
    import gc
    import torch
    
    # Finish wandb run if active
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    
    # Free trainer memory
    if trainer is not None:
        try:
            if hasattr(trainer, "accelerator"):
                trainer.accelerator.free_memory()
            del trainer
        except Exception:
            pass
    
    # Delete model
    if model is not None:
        try:
            del model
        except Exception:
            pass
    
    # Delete extra variables
    if extra_vars:
        for v in extra_vars:
            try:
                del v
            except Exception:
                pass
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    print("Model and caches cleaned up")
