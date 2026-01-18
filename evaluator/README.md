# Human Evaluation Framework

A Next.js application for conducting rigorous human evaluation of model predictions with blinded A/B comparison and structured rubrics.

## Features

- **Model Discovery**: Automatically discovers model folders from the parent directory
- **Stratified Sampling**: Reproducible sampling with configurable seed and percentage (≥5% recommended)
- **Blinded Evaluation**: Predictions labeled A/B/C with randomized order, model identity hidden during rating
- **Dual Evaluation Modes**:
  - **Label Validation**: Binary valid/invalid assessment of data extraction quality
  - **Output Quality Rating**: 1-5 scale across Relevance, Coherence, Naturalness, Specificity
- **Progress Tracking**: Real-time progress percentage across all samples
- **Session Management**: Resume interrupted sessions, multiple raters supported
- **Export**: CSV export per model folder and reproducible `sample_ids.json`

## Getting Started

```bash
# Install dependencies
bun install

# Run development server
bun run dev
```

Open [http://localhost:3000](http://localhost:3000) to start evaluating.

## Evaluation Design

### Sampling (Reproducible)

- **Population**: All samples used in training + evaluation (5,100 total)
- **Sample Size**: Configurable, ≥5% recommended per academic standards
- **Stratification**: By dataset (WildChat vs SGD)
- **Seed**: Fixed random seed (default: 42) for reproducibility

Export sample IDs for reproducibility:

```bash
curl "http://localhost:3000/api/samples/export?samplePercentage=5&seed=42" > sample_ids.json
```

### Rating Categories (1-5 Scale)

| Category        | Description                                                                              |
| --------------- | ---------------------------------------------------------------------------------------- |
| **Relevance**   | Does the user turn directly respond to the assistant's last message and remain on-topic? |
| **Coherence**   | Is it logically consistent with the prior turns and doesn't contradict the conversation? |
| **Naturalness** | Does it sound like a plausible human user message (not assistant-y, not overly verbose)? |
| **Specificity** | (Optional) Does it move the conversation forward with meaningful info/question?          |

### Calibration Examples

Before rating, review these anchor points:

**Relevance**

- 1: Completely off-topic, ignores assistant's message entirely
- 3: Partially relevant, addresses some aspects but drifts
- 5: Directly responds to assistant, stays fully on-topic

**Coherence**

- 1: Contradicts previous statements, logically inconsistent
- 3: Mostly consistent but has minor logical gaps
- 5: Perfectly consistent with all prior context

**Naturalness**

- 1: Sounds like an AI assistant, overly formal or verbose
- 3: Somewhat natural but has awkward phrasing
- 5: Completely natural, indistinguishable from human

### Issue Tags

Optional tags for common problems:

- `assistant-like`: Response sounds like an assistant, not a user
- `hallucinated-constraint`: Introduces constraints not in context
- `topic-drift`: Gradually moves away from conversation topic
- `repetitive`: Repeats previous content unnecessarily
- `truncated`: Response appears cut off
- `grammatical-error`: Contains grammar/spelling issues
- `wrong-language`: Not in expected language
- `too-verbose`: Unnecessarily long
- `too-terse`: Too short to be meaningful

## Output Files

### Per Session

- `data/sessions/{session_id}.json` - Full session state with all ratings

### Per Model (after export)

- `{model_folder}/human_eval_ratings.csv` - Ratings for that model's predictions

### CSV Schema

```csv
sampleId,blindedLabel,modelId,predictionType,relevance,coherence,naturalness,specificity,issueTag,notes,timestamp,raterId
eval-42,A,Qwen/Qwen2.5-3B-Instruct,fineTuned,4,5,4,3,,,2024-01-15T10:30:00Z,rater1
```

## API Endpoints

| Endpoint              | Method   | Description                             |
| --------------------- | -------- | --------------------------------------- |
| `/api/models`         | GET      | List discovered models and corpus stats |
| `/api/samples`        | GET      | Get stratified sample set               |
| `/api/samples/export` | GET      | Export sample IDs for reproducibility   |
| `/api/sessions`       | GET/POST | List or create evaluation sessions      |
| `/api/sessions/[id]`  | GET/POST | Get session or add ratings              |

## Inter-Rater Agreement (Optional)

For stronger credibility, have a second rater evaluate 10-20% of samples and compute:

- Krippendorff's alpha (ideal)
- Weighted Cohen's kappa

## Tech Stack

- Next.js 16 with App Router
- TypeScript
- Tailwind CSS v4
- shadcn/ui components
- PapaParse for CSV handling
