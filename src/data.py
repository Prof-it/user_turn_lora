"""
Data loading and processing module for UserTurnLoRA pipeline.
Handles WildChat and Schema-Guided Dialog datasets.
"""

import hashlib
import random
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset

from .config import PipelineConfig, SYSTEM_PROMPT


def _hash_conversation(conv: List[Dict]) -> str:
    """Create a hash of conversation for deduplication."""
    text = "".join(m.get("content", "") for m in conv)
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _process_wildchat_item(item: Dict) -> Optional[Dict]:
    """
    Process a single WildChat conversation into (context, target_user) pair.
    
    Returns None if the conversation doesn't meet criteria.
    """
    conv = item.get("conversation", [])
    lang = item.get("language", "")
    
    # Filter: English only, minimum 2 turns
    if lang != "English" or len(conv) < 2:
        return None
    
    def _get_text(message: Dict) -> str:
        if message.get("content") is not None:
            return str(message.get("content"))
        if message.get("text") is not None:
            return str(message.get("text"))
        return ""

    def _norm_role(role: Optional[str]) -> str:
        normalized = (role or "").strip().lower()
        if normalized in {"assistant", "system", "tool"}:
            return "assistant"
        if normalized in {"user", "human"}:
            return "user"
        return normalized or "user"

    # Normalize messages and drop extra fields (e.g., timestamps)
    msgs = [
        {
            "role": _norm_role(m.get("role")),
            "content": _get_text(m),
            "language": m.get("language") or lang,
        }
        for m in conv
    ]

    if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None

    # Drop final assistant; target is last user before it
    msgs_wo_last_assistant = msgs[:-1]
    if len(msgs_wo_last_assistant) < 2:
        return None

    target = msgs_wo_last_assistant[-1]
    if target["role"] != "user":
        return None

    context = msgs_wo_last_assistant[:-1]
    if not context or context[-1]["role"] != "assistant":
        return None

    context_clean = [{"role": m["role"], "content": m["content"]} for m in context]
    target_user = target.get("content", "")

    return {
        "conversation": context_clean,
        "target_user": target_user,
        "meta": {
            "dataset": "allenai/WildChat-1M",
            "language": target.get("language") or lang,
            "num_turns": len(context_clean) + 1,
            "conversation_hash": _hash_conversation(context_clean),
        },
    }


def _process_sgd_item(item: Dict, seen_dialogs: set) -> Optional[Dict]:
    """
    Process a single Schema-Guided Dialog item (parquet format).
    
    Parquet format has:
    - context: list of strings (alternating user/assistant)
    - prompt: the target user turn
    - dialog_id: unique dialog identifier
    """
    dialog_id = item.get("dialog_id", "")
    
    # Get context (list of turn strings) and prompt (target user turn)
    context_turns = item.get("context", [])
    target_user = item.get("prompt", "")
    
    # Need at least 2 turns in context (user + assistant)
    if not isinstance(context_turns, list) or len(context_turns) < 2:
        return None
    
    if not target_user:
        return None
    
    # Build conversation: alternating user (even idx) / assistant (odd idx)
    # Even length = ends with assistant (0=user, 1=assistant)
    # Odd length = ends with user
    conv = []
    for i, turn in enumerate(context_turns):
        role = "user" if i % 2 == 0 else "assistant"
        conv.append({"role": role, "content": turn})
    
    # Context must end with assistant (even length)
    if len(context_turns) % 2 != 0:
        return None
    
    return {
        "conversation": conv,
        "target_user": target_user,
        "meta": {
            "dataset": "GEM/schema_guided_dialog",
            "language": "en",
            "num_turns": len(conv) + 1,
            "conversation_hash": _hash_conversation(conv),
            "dialog_id": dialog_id,
        }
    }


def load_wildchat(num_samples: int) -> List[Dict]:
    """Load and process WildChat dataset."""
    print(f"Loading WildChat dataset (target: {num_samples} samples)...")
    
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    
    processed = []
    for item in ds:
        result = _process_wildchat_item(item)
        if result:
            processed.append(result)
            if len(processed) >= num_samples:
                break
    
    print(f"  Loaded {len(processed)} WildChat samples")
    return processed


def load_sgd(num_samples: int) -> List[Dict]:
    """Load and process Schema-Guided Dialog dataset."""
    print(f"Loading Schema-Guided Dialog dataset (target: {num_samples} samples)...")
    
    # Use parquet revision with streaming to avoid downloading full dataset
    ds = load_dataset("GEM/schema_guided_dialog", split="train", revision="refs/convert/parquet", streaming=True)
    
    seen_dialogs = set()
    processed = []
    
    for item in ds:
        result = _process_sgd_item(item, seen_dialogs)
        if result:
            processed.append(result)
            if len(processed) >= num_samples:
                break
    
    print(f"  Loaded {len(processed)} SGD samples")
    return processed


def load_data(config: PipelineConfig, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and split data into training and evaluation sets.
    
    Args:
        config: Pipeline configuration
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (training_pairs, eval_pairs)
    """
    total_samples = config.num_train_samples + config.num_eval_samples
    wildchat_count = int(total_samples * config.wildchat_ratio)
    sgd_count = total_samples - wildchat_count
    
    print(f"\nLoading data: {total_samples} total ({wildchat_count} WildChat, {sgd_count} SGD)")
    
    # Load both datasets
    wildchat_pairs = load_wildchat(wildchat_count)
    sgd_pairs = load_sgd(sgd_count)
    
    # Interleave for balanced distribution (deterministic order)
    all_pairs = []
    for i in range(max(len(wildchat_pairs), len(sgd_pairs))):
        if i < len(wildchat_pairs):
            all_pairs.append(wildchat_pairs[i])
        if i < len(sgd_pairs):
            all_pairs.append(sgd_pairs[i])
    
    # Split into eval and train (no shuffling for reproducibility)
    eval_pairs = all_pairs[:config.num_eval_samples]
    train_pairs = all_pairs[config.num_eval_samples:config.num_eval_samples + config.num_train_samples]
    
    print(f"Split: {len(train_pairs)} training, {len(eval_pairs)} evaluation")
    
    # Print composition
    train_wc = sum(1 for p in train_pairs if "WildChat" in p["meta"]["dataset"])
    train_sgd = len(train_pairs) - train_wc
    eval_wc = sum(1 for p in eval_pairs if "WildChat" in p["meta"]["dataset"])
    eval_sgd = len(eval_pairs) - eval_wc
    
    print(f"  Training: {train_wc} WildChat, {train_sgd} SGD")
    print(f"  Evaluation: {eval_wc} WildChat, {eval_sgd} SGD")
    
    return train_pairs, eval_pairs


def format_for_training(
    pairs: List[Dict],
    tokenizer,
    config: PipelineConfig,
) -> Dataset:
    """
    Format conversation pairs for SFT training.
    
    Creates input_ids and labels where only the target user turn is supervised.
    """
    from .config import SYSTEM_PROMPT
    
    def normalize_conv(conv: List[Dict]) -> List[Dict]:
        """Normalize conversation to standard format."""
        out = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in conv:
            role = m.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            out.append({"role": role, "content": m.get("content", "")})
        return out
    
    def build_training_example(item: Dict) -> Dict:
        """Build a single training example with masked labels."""
        conv = normalize_conv(item["conversation"])
        target = item["target_user"]
        
        # Add target user turn
        conv.append({"role": "user", "content": target})
        
        # Tokenize full conversation
        full_ids = tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=False,
        )
        
        # Tokenize context only (without target)
        context_conv = conv[:-1]
        context_ids = tokenizer.apply_chat_template(
            context_conv,
            tokenize=True,
            add_generation_prompt=False,
        )
        
        # Create labels: -100 for context, actual ids for target
        labels = [-100] * len(context_ids) + full_ids[len(context_ids):]
        
        # Truncate if needed
        if len(full_ids) > config.max_context_len:
            full_ids = full_ids[:config.max_context_len]
            labels = labels[:config.max_context_len]
        
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }
    
    print(f"Formatting {len(pairs)} examples for training...")
    examples = [build_training_example(p) for p in pairs]
    
    return Dataset.from_list(examples)


def build_eval_examples(pairs: List[Dict]) -> List[Dict]:
    """
    Build evaluation examples for perplexity calculation.
    
    Returns list of dicts with conversation and target_user.
    """
    return [
        {
            "conversation": p["conversation"],
            "target_user": p["target_user"],
            "meta": p["meta"],
        }
        for p in pairs
    ]
