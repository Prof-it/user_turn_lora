Multi-turn training note
========================

Why fine-tuning may not fully help multi-turn rollout
----------------------------------------------------
- The current training objective is still single-step, teacher-forced next-user prediction.
- The model sees multi-turn context, but loss is only applied to one gold target user turn.
- During rollout, the model conditions on its own previous outputs, which creates exposure bias.
- This is a plausible reason why a model can improve on single-turn metrics but still degrade or behave inconsistently in multi-turn evaluation.

Important nuance
----------------
- This is probably not the only reason.
- In the current results, LiquidAI, Qwen, and Llama remain strong in rollout even though they were trained with the same one-step objective.
- So for OLMo specifically, the weak rollout behavior is likely a combination of:
  - one-step training objective
  - model-family sensitivity
  - configuration sensitivity
  - rollout calibration/prompting effects
  - single-turn-best config not necessarily being rollout-best

Potential future training change
--------------------------------
- If needed later, add a separate multi-turn-aware training variant instead of rewriting the current pipeline.
- Most sensible options:
  - sample multiple random dialogue prefixes from a trajectory instead of only one fixed target position
  - add self-conditioned / rollout-aware training where the user model sometimes conditions on its own earlier generated turns, with gold assistant turns injected between steps

Decision for current revision
-----------------------------
- Do not change the training pipeline yet.
- Let all current experiments finish first.
- Inspect the complete evidence pack.
- Then decide whether multi-turn-aware training is necessary.
- For the current reviewer feedback, multi-turn evaluation is the essential addition right now; multi-turn training is not required unless the final results clearly demand it.
