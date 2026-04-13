"""
Reference-assisted and free-assistant multi-turn rollout loops.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def _copy_history(messages: List[Dict]) -> List[Dict]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def _build_step_record(
    dialogue: Dict,
    *,
    step_index: int,
    seed_pairs: int,
    history_before_step: List[Dict],
    predicted_user: str,
    gold_user: str,
    gold_assistant: str,
    generated_assistant: Optional[str],
) -> Dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "dataset": dialogue["meta"].get("dataset", ""),
        "conversation_hash": dialogue["meta"].get("conversation_hash", ""),
        "seed_pairs": seed_pairs,
        "step_index": step_index,
        "history_pairs_before_step": len(history_before_step) // 2,
        "predicted_user": predicted_user,
        "gold_user": gold_user,
        "gold_assistant": gold_assistant,
        "generated_assistant": generated_assistant,
    }


def _build_summary(dialogue: Dict, *, steps: List[Dict], stop_reason: str) -> Dict:
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "dataset": dialogue["meta"].get("dataset", ""),
        "service": dialogue["meta"].get("service", ""),
        "num_pairs": dialogue["meta"].get("num_pairs", len(dialogue["conversation"]) // 2),
        "completed_steps": len(steps),
        "stop_reason": stop_reason,
    }


def run_reference_assisted_rollout(
    dialogue: Dict,
    *,
    user_generate: Callable[[List[Dict]], str],
    seed_pairs: int,
    max_rollout_steps: int,
) -> Dict:
    """Advance the dialogue with predicted user turns and gold assistant replies."""
    conversation = dialogue["conversation"]
    history = _copy_history(conversation[: seed_pairs * 2])
    predicted_user_history: List[str] = []
    step_records: List[Dict] = []
    stop_reason = "max_steps"

    for step_index in range(max_rollout_steps):
        pair_offset = seed_pairs + step_index
        gold_user_index = pair_offset * 2
        gold_assistant_index = gold_user_index + 1
        if gold_assistant_index >= len(conversation):
            stop_reason = "dialogue_exhausted"
            break

        gold_user = conversation[gold_user_index]["content"]
        gold_assistant = conversation[gold_assistant_index]["content"]
        history_before_step = _copy_history(history)
        predicted_user = user_generate(history_before_step).strip()
        record = _build_step_record(
            dialogue,
            step_index=step_index,
            seed_pairs=seed_pairs,
            history_before_step=history_before_step,
            predicted_user=predicted_user,
            gold_user=gold_user,
            gold_assistant=gold_assistant,
            generated_assistant=None,
        )
        step_records.append(record)

        normalized_prediction = _normalize_text(predicted_user)
        if not normalized_prediction:
            stop_reason = "empty_user"
            break
        if normalized_prediction in predicted_user_history:
            stop_reason = "repeated_user"
            break

        predicted_user_history.append(normalized_prediction)
        history.append({"role": "user", "content": predicted_user})
        history.append({"role": "assistant", "content": gold_assistant})

    return {
        "dialogue": _build_summary(dialogue, steps=step_records, stop_reason=stop_reason),
        "steps": step_records,
    }


def run_free_assistant_rollout(
    dialogue: Dict,
    *,
    user_generate: Callable[[List[Dict]], str],
    assistant_generate: Callable[[List[Dict]], str],
    seed_pairs: int,
    max_rollout_steps: int,
) -> Dict:
    """Advance the dialogue with predicted user turns and generated assistant replies."""
    conversation = dialogue["conversation"]
    history = _copy_history(conversation[: seed_pairs * 2])
    predicted_user_history: List[str] = []
    step_records: List[Dict] = []
    stop_reason = "max_steps"

    for step_index in range(max_rollout_steps):
        pair_offset = seed_pairs + step_index
        gold_user_index = pair_offset * 2
        gold_assistant_index = gold_user_index + 1
        if gold_assistant_index >= len(conversation):
            stop_reason = "dialogue_exhausted"
            break

        gold_user = conversation[gold_user_index]["content"]
        gold_assistant = conversation[gold_assistant_index]["content"]
        history_before_step = _copy_history(history)
        predicted_user = user_generate(history_before_step).strip()
        normalized_prediction = _normalize_text(predicted_user)
        generated_assistant = ""

        if not normalized_prediction:
            stop_reason = "empty_user"
        elif normalized_prediction in predicted_user_history:
            stop_reason = "repeated_user"
        else:
            predicted_user_history.append(normalized_prediction)
            history.append({"role": "user", "content": predicted_user})
            generated_assistant = assistant_generate(_copy_history(history)).strip()
            if not _normalize_text(generated_assistant):
                stop_reason = "empty_assistant"
            else:
                history.append({"role": "assistant", "content": generated_assistant})

        record = _build_step_record(
            dialogue,
            step_index=step_index,
            seed_pairs=seed_pairs,
            history_before_step=history_before_step,
            predicted_user=predicted_user,
            gold_user=gold_user,
            gold_assistant=gold_assistant,
            generated_assistant=generated_assistant or None,
        )
        step_records.append(record)

        if stop_reason != "max_steps":
            break

    return {
        "dialogue": _build_summary(dialogue, steps=step_records, stop_reason=stop_reason),
        "steps": step_records,
    }
