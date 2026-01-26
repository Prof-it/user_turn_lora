# User Turn LoRA

Fine-tuning LLMs to predict user turns in conversations.

## Model Performance Overview

![Cross-Model Comparison](cross_model_comparison.png)

## Project Structure

```
├── create_plots.py          # Main script to generate thesis figures
├── modules/
│   ├── plot1.py             # Benchmark comparison (Base vs Fine-tuned)
│   ├── plot2.py             # Domain-specific analysis
│   └── helpers.py           # Utility functions
├── outputs/                  # Model evaluation results
│   ├── Qwen-Qwen2.5-3B-Instruct/
│   ├── LiquidAI-LFM2.5-1.2B-Instruct/
│   ├── allenai-OLMo-3-7B-Instruct/
│   └── ablation/            # Ablation study results
├── notebook/
│   └── UserTurnLoRA.ipynb   # Main experiment notebook
└── src/                     # Training and evaluation source code
```

## Generating Plots

```bash
# Process all discovered models
python create_plots.py

# Process specific model directory
python create_plots.py --model-dir outputs/Qwen-Qwen2.5-3B-Instruct

# List available models
python create_plots.py --list-models

# Quiet mode (suppress verbose output)
python create_plots.py --quiet
```

## Environment

Successfully run on **Google Colab A100 GPU**.

> ⚠️ **Note**: Tests on Colab H100 and RunPod H100 failed due to BLEURT library compatibility issues.

## Notes

- Any Hugging Face Transformers text model that supports the `apply_chat_template` method can be used in this pipeline.

---

## Methodology Details

### Data Sources & Composition

![Dataset Split](dataset_split.png)

| Dataset                                                                              | Domain        | Samples   | Proportion |
| ------------------------------------------------------------------------------------ | ------------- | --------- | ---------- |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)           | Open-domain   | 2,550     | 50%        |
| [GEM/schema_guided_dialog](https://huggingface.co/datasets/GEM/schema_guided_dialog) | Task-oriented | 2,550     | 50%        |
| **Total**                                                                            |               | **5,100** | 100%       |

**Split:**

- Training: 5,000 samples (2,500 per dataset)
- Evaluation: 100 samples (50 per dataset)

### Labeling Procedure

**Automatic extraction from conversation structure** — no manual annotation required.

Each sample is constructed as a `(context, target_user)` pair:

1. **Context**: The conversation history ending with an assistant turn
2. **Target**: The next user turn (ground truth for prediction)

**Processing logic** (from `UserTurnLoRA.ipynb`):

```
Conversation: [User₁, Assistant₁, User₂, Assistant₂, User₃, Assistant₃]
                ↓
Context:      [User₁, Assistant₁, User₂, Assistant₂]  (ends with assistant)
Target:       User₃                                    (next user turn)
```

**Filtering criteria:**

- Minimum 2 turns per conversation
- WildChat: English language only
- SGD: Deduplicated by `dialog_id` (keeps latest turn per dialog)

### Metrics Explanation

| Metric           | Measures                                       | Direction          | Range          | Human Correlation                                                     |
| ---------------- | ---------------------------------------------- | ------------------ | -------------- | --------------------------------------------------------------------- |
| **Perplexity**   | Model uncertainty (effective branching factor) | ↓ Lower is better  | Model-specific | N/A — measures confidence, not accuracy                               |
| **BERTScore-F1** | Semantic similarity via contextual embeddings  | ↑ Higher is better | -1 to 1        | 0.93 Pearson ([Zhang et al., 2020](https://arxiv.org/abs/1904.09675)) |
| **BLEURT**       | Learned metric trained on human judgments      | ↑ Higher is better | ~0 to 1        | Trained on WMT human ratings                                          |

> **Note:** These metrics have **no universal "good" thresholds** — they are designed for _relative comparison_, not absolute judgment. Perplexity depends on tokenization/vocabulary; BERTScore depends on the underlying BERT model. This is why we report **Δ (delta)** rather than absolute values.

### Why Report Δ (Delta)?

Reporting relative change isolates the fine-tuning effect and is the methodologically correct approach:

- **Controls for model-specific factors** (tokenizer, vocabulary, architecture)
- **Enables cross-model comparison** (Qwen vs LiquidAI have different baselines)
- **Shows improvement** attributable to fine-tuning, not absolute capability

_Example:_ Perplexity Δ = -40% → fine-tuned model considers 40% fewer plausible options per token (more confident predictions)
