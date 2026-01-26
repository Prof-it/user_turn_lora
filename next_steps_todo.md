# SDS2026 Paper – Next Steps To‑Do (LoRA User‑Turn Prediction)

This checklist is **non-technical** (no implementation details). It’s meant to be pasted into the repo and used as a working plan.

## Goal (paper scope)

Show evidence that **LoRA fine‑tuning** improves **next user turn prediction** on domain-specific dialogues versus:

1. **Base model** (no fine‑tuning)
2. **Prompt-only / zero‑shot (or few‑shot)** baseline (API model / prompting)

Use: **Perplexity (absolute + Δ), BERTScore, BLEURT**, plus a small **human rating study**.

---

## 0) Admin / Setup

- [x] Confirm target venue + format (SDS2026 short paper, ~6–8 pages).
- [x] Use the Overleaf project and start the paper skeleton (intro/method/results placeholders).
- [x] Make sure any API keys are **never committed** (store only in local env/secrets).
- [x] Create modular .py files (not notebooks) for reproducibility and Docker deployment.
- [x] Set up Docker + GCloud deployment pipeline (A100 VM).
- [x] Document everything so results are saved and no re-runs needed.

---

## 1) Freeze experimental configuration (post‑ablation)

You already ran an ablation on one model.

- [x] Select **best configuration** from the ablation (define the selection criterion).
- [x] Lock this configuration as the **final config** to reuse for other model families.
- [x] Record the final config in one place (README / config file) so it's unambiguous.

Deliverable:

- [x] A short "Final Training Config" note (what was chosen and why). → See `paper/ablation.md` and `outputs/ablation_paper/optimal_config.json`

---

## 2) Run core experiments on all chosen models (Base vs LoRA)

Minimal requirement: run on the same evaluation subset for clean comparisons.

- [x] Choose evaluation subset size for **metric evaluation**:
  - Used: 400 samples (6.25% of 6400 total)
- [x] For each model family:
  - [x] Base model predictions (no LoRA)
  - [x] LoRA fine‑tuned predictions (final config)

**Completed 2026-01-23**: All 4 models trained and evaluated:

- LiquidAI/LFM2.5-1.2B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- allenai/OLMo-3-7B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

Deliverables:

- [x] Saved predictions per model and condition (base vs LoRA)
- [x] Metric outputs per model and condition (PPL abs + Δ, BERTScore, BLEURT)

### 2.1 Temperature Sweep (completed 2026-01-25)

- [x] Run temperature sweep (0.3, 0.4, 0.5, 0.6, 0.7) on all 4 models
- [x] Generate temperature sweep plots and LaTeX table
- [x] Document results in `outputs/temperature_sweep_results.md`
- [x] LaTeX table saved to `paper/temperature_sweep_table.tex`

---

## 3) Prompt-only baseline (API model, no fine-tuning)

Objective: answer **"How well can we do without any training, just by prompting a strong model?"**

This baseline uses an **API model (e.g., OpenAI GPT-4o)** with **no fine-tuning** — only prompting.

### 3.1 Setup

- [ ] Select API model (recommended: `gpt-4o` or `gpt-4o-mini` for cost)
- [ ] Create a reusable inference script (`src/prompt_baseline.py`)

### 3.2 Prompt conditions

- [ ] **Zero-shot**: dialogue context + instruction only, no examples
- [ ] **Few-shot (1–3 examples)**: include 1–3 exemplar (context → user turn) pairs in the prompt

Prompt structure (example):

```
System: You are predicting what a human user would say next in a conversation.

[Optional few-shot examples here]

Dialogue context:
{context}

Predict the next user turn:
```

### 3.3 Run inference

- [ ] Generate predictions on the **same metric evaluation subset** used for Base/LoRA
- [ ] Run both zero-shot and few-shot conditions
- [ ] Log token usage / cost for reproducibility

### 3.4 Compute metrics

- [ ] Compute **same metrics** as other conditions:
  - BERTScore
  - BLEURT
  - (PPL not directly comparable since API models don't expose logprobs the same way)

Deliverables:

- [ ] `prompts/zero_shot.txt` and `prompts/few_shot.txt` — fixed prompt templates
- [ ] `outputs/openai/gpt-4o-zero-shot/predictions.json`
- [ ] `outputs/openai/gpt-4o-few-shot/predictions.json`
- [ ] Metric results for prompt-only conditions

Notes:

- Keep prompts **identical** across samples (only the dialogue context changes).
- Document exact model name + version (e.g., `gpt-4o-2024-08-06`).
- Few-shot exemplars should be **fixed** (same examples for all samples) and drawn from **train set only**.

---

## 4) Human evaluation (manual rating)

Per supervisor: **≥5% coverage of full corpus** (~250 items), can be sampled from anywhere (not limited to metric eval split). Single rater is OK; note as limitation.

### 4.1 Sampling (reproducible)

- [ ] Randomly sample **≈250** items from full corpus (both domains), with a fixed seed.
- [ ] Keep a clear record of sample IDs and their domain/source.

### 4.2 Blind rating setup

- [ ] Use the Next.js UI to rate blindly (hide which output is base/LoRA/prompt-only).
- [ ] Rate at least these criteria (1–5):
  - [ ] Relevance
  - [ ] Coherence
  - [ ] Naturalness
- [ ] Optional: add a free-text “issue tag” field (e.g., off-topic, assistant-like, generic, contradiction).

### 4.3 Outputs

- [ ] Export human ratings to CSV (audit-ready).
- [ ] Summarize:
  - [ ] Mean/median per criterion
  - [ ] Base vs LoRA “win rate” (how often LoRA > base)
  - [ ] Breakdown by domain (WildChat vs SGD if applicable)

### 4.4 Limitations

- [ ] Add a paper note: single rater; optional second rater would strengthen reliability.

Deliverables:

- [ ] `human_eval_sample_ids.csv`
- [ ] `human_eval_ratings.csv`
- [ ] 2–3 worked examples for the appendix/README (context → outputs → ratings)

---

## 5) Update tables & plots (paper-ready)

- [ ] Final metrics table per model:
  - PPL (absolute) + Δ
  - BERTScore
  - BLEURT
- [ ] Baseline comparison table:
  - Prompt-only vs Base vs LoRA
- [ ] Human evaluation summary table (and optionally 1 small plot).

Deliverables:

- [ ] A single “Results Index” section in the repo pointing to all final tables/plots.

---

## 6) Draft the short paper (6–8 pages)

Suggested structure:

- [ ] Abstract
- [ ] Introduction (problem + motivation + contributions)
- [ ] Related Work (anchor: Flipping the Dialogue + user LM literature)
- [ ] Method (data, labeling, LoRA setup, prompt-only baseline)
- [ ] Metrics (PPL abs + Δ, BERTScore, BLEURT; what they mean)
- [ ] Human evaluation protocol (sampling, criteria, blindness, limitation)
- [ ] Results + discussion (per model + domain)
- [ ] Limitations + future work
- [ ] Conclusion

Deliverable:

- [ ] First draft with placeholders filled + final numbers inserted.

---

## 7) Reproducibility & artifacts

- [x] Ensure notebooks/scripts run end-to-end in Colab with pinned dependencies.
- [x] Provide W&B links (or exported logs) for key runs.
- [x] Ensure all outputs (predictions/metrics/human eval CSVs) are versioned and discoverable.
- [x] Add a short "How to reproduce" section. → See `paper/ablation.md`

---

## 8) Plotting & Visualization

- [x] Use tueplots for conference-ready plots: https://github.com/pnkraemer/tueplots
- [x] Generate ablation heatmaps and training curves
- [x] Generate cross-model comparison plots
- [x] Generate temperature sweep plots and heatmap

---

## Quick “definition of done” (submission readiness)

- [ ] Metrics computed for Base vs LoRA vs Prompt-only on a shared eval subset.
- [ ] Human evaluation completed on ≥250 random corpus samples, with blind ratings and CSV export.
- [ ] Final tables/plots and a results index exist.
- [ ] Overleaf draft is complete (6–8 pages) and cites Flipping the Dialogue appropriately.
