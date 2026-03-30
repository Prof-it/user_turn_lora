export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ConversationMeta {
  dataset: string;
  conversation_hash?: string;
  dialog_id?: string;
  language?: string;
  num_turns?: number;
}

export interface ConversationPair {
  conversation: Message[];
  target_user: string;
  meta: ConversationMeta;
}

export interface PredictionRow {
  ref: string;
  pred?: string;
  bertscore_f1?: number;
  bleurt?: number;
  ppl_content?: number;
}

export interface ConditionMetrics {
  bertscore_f1?: number | null;
  bleurt?: number | null;
  ppl_content?: number | null;
}

export interface PredictionCondition {
  conditionId: string;
  displayName: string;
  text: string;
  metrics: ConditionMetrics;
}

export interface ModelConfig {
  model_name: string;
  chat_pairs: number;
  training_pairs: number;
  do_sample: boolean;
  config: {
    min_turns: number;
    wildchat_languages: string[];
  };
  NUM_CONVS_TO_PROCESS: number;
  NUM_CONV_FOR_TRAINING: number;
}

export interface ModelData {
  id: string;
  name: string;
  path: string;
  config: ModelConfig;
}

export interface EvaluationSample {
  id: string;
  sampleIndex: number;
  dataset: string;
  context: Message[];
  groundTruth: string;
  predictions: PredictionCondition[];
  // Blinded labels (A/B/C...) mapped to condition IDs - hidden from UI
  blindedMapping: Record<string, string>;
}

export interface RatingCategory {
  id: string;
  name: string;
  description: string;
  examples: {
    score1: string;
    score3: string;
    score5: string;
  };
}

export interface Rating {
  sampleId: string;
  blindedLabel: string; // A, B, C...
  modelId: string; // Revealed after rating
  predictionType: string;
  relevance: number; // 1-5
  coherence: number; // 1-5
  naturalness: number; // 1-5
  specificity?: number; // 1-5 (optional)
  issueTag?: string;
  notes?: string;
  timestamp: string;
  raterId: string;
}

export interface LabelValidation {
  sampleId: string;
  isValid: boolean;
  errorType?: "wrong_extraction" | "truncated" | "noise" | "other";
  notes?: string;
  timestamp: string;
  raterId: string;
}

export interface EvaluationSession {
  id: string;
  raterId: string;
  selectedModels: string[];
  sampleIds: string[];
  seed: number;
  startedAt: string;
  ratings: Rating[];
  labelValidations: LabelValidation[];
  progress: {
    completed: number;
    total: number;
  };
}

export interface SamplingConfig {
  seed: number;
  samplePercentage: number;
  stratifyBy: ("dataset" | "turnLength")[];
  turnLengthBuckets: {
    short: [number, number];
    medium: [number, number];
    long: [number, number];
  };
}

export const DEFAULT_SAMPLING_CONFIG: SamplingConfig = {
  seed: 42,
  samplePercentage: 5,
  stratifyBy: ["dataset"],
  turnLengthBuckets: {
    short: [2, 4],
    medium: [5, 8],
    long: [9, Infinity],
  },
};

export const RATING_CATEGORIES: RatingCategory[] = [
  {
    id: "relevance",
    name: "Relevance",
    description: "Does the user turn directly respond to the assistant's last message and remain on-topic?",
    examples: {
      score1: "Completely off-topic, ignores assistant's message entirely",
      score3: "Partially relevant, addresses some aspects but drifts",
      score5: "Directly responds to assistant, stays fully on-topic",
    },
  },
  {
    id: "coherence",
    name: "Coherence",
    description: "Is it logically consistent with the prior turns and doesn't contradict the conversation?",
    examples: {
      score1: "Contradicts previous statements, logically inconsistent",
      score3: "Mostly consistent but has minor logical gaps",
      score5: "Perfectly consistent with all prior context",
    },
  },
  {
    id: "naturalness",
    name: "Naturalness",
    description: "Does it sound like a plausible human user message (not assistant-y, not overly verbose)?",
    examples: {
      score1: "Sounds like an AI assistant, overly formal or verbose",
      score3: "Somewhat natural but has awkward phrasing",
      score5: "Completely natural, indistinguishable from human",
    },
  },
  {
    id: "specificity",
    name: "Specificity/Usefulness",
    description: "Does it move the conversation forward with meaningful info/question (not generic fluff)?",
    examples: {
      score1: "Generic filler like 'okay' or 'I see' with no substance",
      score3: "Some useful content but could be more specific",
      score5: "Provides specific, actionable information or question",
    },
  },
];

export const ISSUE_TAGS = [
  "assistant-like",
  "hallucinated-constraint",
  "topic-drift",
  "repetitive",
  "truncated",
  "grammatical-error",
  "wrong-language",
  "too-verbose",
  "too-terse",
  "other",
] as const;

export type IssueTag = (typeof ISSUE_TAGS)[number];
