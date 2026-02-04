# Human Evaluation Protocol & Results

## Overview

Human evaluation complements automated metrics (BERTScore, BLEURT, PPL) by assessing **intrinsic quality** of generated user turns.

| Metric        | Value                                |
| ------------- | ------------------------------------ |
| Samples rated | 369                                  |
| Total corpus  | 6,400 (400 eval + 6,000 train)       |
| Coverage      | **5.77%** (target: >5%) ✓            |
| Rater         | Single rater (author)                |
| Blinding      | Randomized labels A/B/C/D per sample |

## Conditions Evaluated

Human evaluation uses **Qwen2.5-3B-Instruct** as the representative model:

| Condition  | Model                      | Description                            |
| ---------- | -------------------------- | -------------------------------------- |
| Baseline   | Qwen2.5-3B-Instruct        | No fine-tuning                         |
| Fine-tuned | Qwen2.5-3B-Instruct + LoRA | Optimal config from ablation           |
| Zero-shot  | GPT-4o-mini                | Instruction only, no examples          |
| Few-shot   | GPT-4o-mini                | 3 exemplar (context → user turn) pairs |

**Note:** Automated metrics cover all 4 model families (Qwen, Llama, OLMo, LiquidAI). Human evaluation focuses on one representative model to keep annotation tractable.

## Rating Categories (1-5 scale)

### 1. Relevance

Does the predicted user turn directly respond to what the assistant just said?

| Score | Description                                                    |
| ----- | -------------------------------------------------------------- |
| 1     | Completely off-topic, ignores the assistant's message entirely |
| 2     | Loosely related but misses the main point                      |
| 3     | Partially relevant, addresses some aspects but drifts          |
| 4     | Mostly relevant, minor tangents                                |
| 5     | Directly responds to the assistant, stays fully on-topic       |

### 2. Coherence

Is it logically consistent with the conversation history?

| Score | Description                                       |
| ----- | ------------------------------------------------- |
| 1     | Contradicts previous statements, logically broken |
| 2     | Major inconsistencies with prior context          |
| 3     | Mostly consistent but has minor logical gaps      |
| 4     | Consistent, only tiny issues                      |
| 5     | Perfectly consistent with all prior context       |

### 3. Naturalness

Does it sound like something a real human user would type?

| Score | Description                                                   |
| ----- | ------------------------------------------------------------- |
| 1     | Sounds like an AI assistant (overly helpful, formal, verbose) |
| 2     | Awkward phrasing, unnatural structure                         |
| 3     | Somewhat natural but has odd word choices                     |
| 4     | Natural with minor quirks                                     |
| 5     | Completely natural, indistinguishable from a real user        |

## Evaluation Guidelines

### Ground Truth Usage

- Ground truth is provided as a **reference only** to understand context
- Predictions are rated on **intrinsic quality**, not similarity to ground truth
- A prediction that differs from ground truth can still score 5/5 if contextually appropriate
- Automated metrics (BERTScore, BLEURT) already capture similarity to ground truth

### Key Principles

1. **Rate each prediction independently** - do not compare A vs B, rate each on its own merit
2. **Don't penalize for different content** - multiple valid responses exist for any conversation
3. **Watch for "assistant-like" language** - common failure mode where prediction sounds like an AI, not a user
4. **Consider the conversation flow** - would this be a reasonable thing for a user to say here?

## Results

### Mean Scores by Condition (1-5 scale)

| Condition      | Relevance | Coherence | Naturalness | **Average** |
| -------------- | --------- | --------- | ----------- | ----------- |
| Baseline       | 4.34      | 4.34      | 4.44        | **4.37**    |
| **Fine-tuned** | 4.64      | 4.68      | 4.81        | **4.71**    |
| Zero-shot      | 4.78      | 4.82      | 4.85        | **4.82**    |
| Few-shot       | 4.77      | 4.80      | 4.83        | **4.80**    |

### Key Findings

1. **Fine-tuned > Baseline**: +0.34 average improvement (4.71 vs 4.37)
2. **Prompt baselines lead**: GPT-4o-mini (4.80-4.82) outperforms Qwen 3B models (expected given model size)
3. **Naturalness shows largest gain**: Fine-tuned +0.37 over baseline (4.81 vs 4.44)
4. **High tie rate (93.8%)**: Most samples rated equally across conditions

### Results by Dataset

| Dataset                      | Baseline | Fine-tuned | Δ         | Zero-shot | Few-shot |
| ---------------------------- | -------- | ---------- | --------- | --------- | -------- |
| Schema-Guided Dialog (n=200) | 4.50     | 4.95       | **+0.45** | 4.94      | 4.99     |
| WildChat (n=169)             | 4.22     | 4.43       | **+0.21** | 4.67      | 4.58     |

**Observation:** Fine-tuning shows larger gains on task-oriented dialog (Schema-Guided) than open-domain chat (WildChat).

### Win Rates

| Winner     | Count | Percentage |
| ---------- | ----- | ---------- |
| Tie        | 346   | 93.8%      |
| Baseline   | 7     | 1.9%       |
| Fine-tuned | 7     | 1.9%       |
| Zero-shot  | 6     | 1.6%       |
| Few-shot   | 3     | 0.8%       |

## Sampling

- **Seed**: 42 (reproducible)
- **Dataset distribution**: 54% Schema-Guided Dialog, 46% WildChat
- **Alignment**: Predictions matched by ground truth text (not index)

## Output

- `outputs/Qwen-Qwen2.5-3B-Instruct/human_eval_ratings.csv`
- Columns: sample*id, dataset, {condition}*{category}, winner, timestamp

## Limitations

- Single rater (inter-rater reliability not measured)
- Subjective assessment (mitigated by rubrics and blinding)
- Representative model only (Qwen); automated metrics cover all 4 families
