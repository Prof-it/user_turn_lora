"""
Generic chat-generation helpers shared across evaluation modes.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import LogitsProcessorList
from transformers.generation.logits_process import NoBadWordsLogitsProcessor


def _build_bad_words(tokenizer, blocked_texts: List[str]) -> List[List[int]]:
    """Tokenize role headers or other blocked strings for generation control."""
    token_ids = tokenizer(blocked_texts, add_special_tokens=False, return_tensors="pt")["input_ids"].tolist()
    return [ids for ids in token_ids if ids]


@torch.no_grad()
def generate_from_messages(
    model,
    tokenizer,
    messages: List[dict],
    config,
    blocked_texts: Optional[List[str]] = None,
    verbose: bool = False,
) -> str:
    """Generate a response from arbitrary chat messages using model config."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    logits_processor = None
    if blocked_texts:
        bad_words = _build_bad_words(tokenizer, blocked_texts)
        if bad_words:
            logits_processor = LogitsProcessorList(
                [NoBadWordsLogitsProcessor(bad_words, eos_token_id=tokenizer.eos_token_id)]
            )

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        no_repeat_ngram_size=4,
        logits_processor=logits_processor,
    )

    gen_ids = outputs[0, input_ids.shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    if verbose:
        print(f"Generated: {text[:200]}...")
    return text
