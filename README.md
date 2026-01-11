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
├── Qwen/
│   └── Qwen2.5-3B-Instruct/  # Model evaluation results
│       ├── eval_bleurt_bertscore_summary.csv
│       ├── eval_ft_bleurt_bertscore_summary.csv
│       ├── chat_pairs.json
│       ├── *.png             # Generated plots
│       └── wandb_screenshot.png  # WandB run report
└── UserTurnLoRA.ipynb       # Main experiment notebook
```

## Generating Plots

```bash
# Process all discovered models
python create_plots.py

# Process specific model directory
python create_plots.py --model-dir Qwen/Qwen2.5-3B-Instruct

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

## TODO

- [ ] Debug BLEURT library failures on H100 GPUs (Colab H100, RunPod H100)
- [ ] Add support for additional models
- [ ] Run with increased sample size
