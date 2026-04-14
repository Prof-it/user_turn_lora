"""
Dialogue trajectory loaders for multi-turn rollout evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from datasets import load_dataset

from .data import _hash_conversation


def _normalize_role(role: Optional[str]) -> str:
    normalized = (role or "").strip().lower()
    if normalized in {"assistant", "system", "tool"}:
        return "assistant"
    if normalized in {"user", "human"}:
        return "user"
    return normalized or "user"


def _extract_text(message: Dict) -> str:
    if message.get("content") is not None:
        return str(message["content"])
    if message.get("text") is not None:
        return str(message["text"])
    return ""


def _is_strict_user_assistant_dialogue(conversation: List[Dict]) -> bool:
    if len(conversation) < 4 or len(conversation) % 2 != 0:
        return False
    for index, message in enumerate(conversation):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role or not message["content"].strip():
            return False
    return conversation[0]["role"] == "user" and conversation[-1]["role"] == "assistant"


def _build_wildchat_dialogue(item: Dict) -> Optional[Dict]:
    if item.get("language") != "English":
        return None

    conversation = [
        {
            "role": _normalize_role(message.get("role")),
            "content": _extract_text(message).strip(),
        }
        for message in item.get("conversation", [])
    ]
    if not _is_strict_user_assistant_dialogue(conversation):
        return None

    return {
        "dialogue_id": item.get("conversation_hash") or _hash_conversation(conversation),
        "conversation": conversation,
        "meta": {
            "dataset": "allenai/WildChat-1M",
            "language": item.get("language", "English"),
            "num_pairs": len(conversation) // 2,
            "conversation_hash": item.get("conversation_hash") or _hash_conversation(conversation),
        },
    }


def load_wildchat_dialogues(num_dialogues: int) -> List[Dict]:
    """Load full WildChat dialogues for rollout evaluation."""
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    dialogues: List[Dict] = []
    for item in dataset:
        dialogue = _build_wildchat_dialogue(item)
        if dialogue:
            dialogues.append(dialogue)
        if len(dialogues) >= num_dialogues:
            break
    return dialogues


def _validate_sgd_context(conversation: List[Dict], row: Dict) -> bool:
    expected_context = [message["content"] for message in conversation]
    actual_context = [str(text).strip() for text in row.get("context", [])]
    return expected_context == actual_context


def _finalize_sgd_dialogue(dialog_id: str, rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None

    ordered_rows = sorted(rows, key=lambda row: int(row.get("turn_id", 0)))
    conversation: List[Dict] = []
    for row in ordered_rows:
        prompt = str(row.get("prompt") or "").strip()
        target = str(row.get("target") or "").strip()
        if not prompt or not target or not _validate_sgd_context(conversation, row):
            return None
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": target})

    if not _is_strict_user_assistant_dialogue(conversation):
        return None

    return {
        "dialogue_id": dialog_id,
        "conversation": conversation,
        "meta": {
            "dataset": "GEM/schema_guided_dialog",
            "language": "en",
            "service": ordered_rows[0].get("service", ""),
            "num_pairs": len(conversation) // 2,
            "conversation_hash": _hash_conversation(conversation),
        },
    }


def _iter_sgd_dialogues() -> Iterable[Dict]:
    dataset = load_dataset(
        "GEM/schema_guided_dialog",
        split="train",
        revision="refs/convert/parquet",
        streaming=True,
    )
    current_dialog_id: Optional[str] = None
    current_rows: List[Dict] = []

    for row in dataset:
        dialog_id = str(row.get("dialog_id") or "")
        if current_dialog_id is None:
            current_dialog_id = dialog_id
        if dialog_id != current_dialog_id:
            dialogue = _finalize_sgd_dialogue(current_dialog_id, current_rows)
            if dialogue:
                yield dialogue
            current_dialog_id = dialog_id
            current_rows = []
        current_rows.append(row)

    if current_dialog_id is not None:
        dialogue = _finalize_sgd_dialogue(current_dialog_id, current_rows)
        if dialogue:
            yield dialogue


def load_sgd_dialogues(num_dialogues: int) -> List[Dict]:
    """Load reconstructed SGD dialogues for rollout evaluation."""
    dialogues: List[Dict] = []
    for dialogue in _iter_sgd_dialogues():
        dialogues.append(dialogue)
        if len(dialogues) >= num_dialogues:
            break
    return dialogues


def filter_dialogues_for_rollout(
    dialogues: List[Dict],
    *,
    seed_pairs: int,
    min_rollout_steps: int,
    max_rollout_steps: Optional[int] = None,
) -> List[Dict]:
    """Keep only dialogues with enough remaining user turns for rollout."""
    filtered = []
    for dialogue in dialogues:
        available_steps = (len(dialogue["conversation"]) // 2) - seed_pairs
        if max_rollout_steps is not None:
            available_steps = min(available_steps, max_rollout_steps)
        if available_steps >= min_rollout_steps:
            filtered.append(dialogue)
    return filtered


def _load_rollout_ready(
    load_dialogues: Callable[[int], List[Dict]],
    num_dialogues: int,
    *,
    seed_pairs: int,
    min_rollout_steps: int,
    max_rollout_steps: Optional[int] = None,
) -> List[Dict]:
    """Collect enough dialogues after applying rollout-length constraints."""
    best: List[Dict] = []
    for multiplier in (1, 2, 4, 8):
        candidates = load_dialogues(num_dialogues * multiplier)
        ready = filter_dialogues_for_rollout(
            candidates,
            seed_pairs=seed_pairs,
            min_rollout_steps=min_rollout_steps,
            max_rollout_steps=max_rollout_steps,
        )
        if len(ready) > len(best):
            best = ready
        if len(ready) >= num_dialogues:
            return ready[:num_dialogues]
    return best[:num_dialogues]


def load_rollout_dialogues(
    dataset_name: str,
    *,
    num_dialogues: int,
    seed_pairs: int,
    min_rollout_steps: int,
    max_rollout_steps: Optional[int] = None,
    wildchat_ratio: float = 0.5,
) -> List[Dict]:
    """Load rollout-ready dialogues from one dataset or a balanced mix."""
    if dataset_name == "wildchat":
        return _load_rollout_ready(
            load_wildchat_dialogues,
            num_dialogues,
            seed_pairs=seed_pairs,
            min_rollout_steps=min_rollout_steps,
            max_rollout_steps=max_rollout_steps,
        )
    elif dataset_name == "sgd":
        return _load_rollout_ready(
            load_sgd_dialogues,
            num_dialogues,
            seed_pairs=seed_pairs,
            min_rollout_steps=min_rollout_steps,
            max_rollout_steps=max_rollout_steps,
        )
    elif dataset_name == "all":
        wildchat_count = max(1, int(round(num_dialogues * wildchat_ratio)))
        sgd_count = max(1, num_dialogues - wildchat_count)
        dialogues = []
        wc = _load_rollout_ready(
            load_wildchat_dialogues,
            wildchat_count,
            seed_pairs=seed_pairs,
            min_rollout_steps=min_rollout_steps,
            max_rollout_steps=max_rollout_steps,
        )
        sgd = _load_rollout_ready(
            load_sgd_dialogues,
            sgd_count,
            seed_pairs=seed_pairs,
            min_rollout_steps=min_rollout_steps,
            max_rollout_steps=max_rollout_steps,
        )
        for index in range(max(len(wc), len(sgd))):
            if index < len(wc):
                dialogues.append(wc[index])
            if index < len(sgd):
                dialogues.append(sgd[index])
        return dialogues[:num_dialogues]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
