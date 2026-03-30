import fs from "fs";
import path from "path";
import Papa from "papaparse";
import {
  ConditionMetrics,
  ConversationPair,
  DEFAULT_SAMPLING_CONFIG,
  EvaluationSample,
  ModelConfig,
  ModelData,
  PredictionCondition,
  PredictionRow,
  SamplingConfig,
} from "@/types/evaluation";

const ROOT_DIR = path.resolve(process.cwd(), "..");

function seededRandom(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function shuffleArray<T>(array: T[], random: () => number): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

export function discoverModels(): ModelData[] {
  const models: ModelData[] = [];
  const entries = fs.readdirSync(ROOT_DIR, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name.startsWith(".") || entry.name === "evaluator" || entry.name === "node_modules") continue;

    const orgPath = path.join(ROOT_DIR, entry.name);
    const subEntries = fs.readdirSync(orgPath, { withFileTypes: true });

    for (const subEntry of subEntries) {
      if (!subEntry.isDirectory()) continue;
      if (subEntry.name.startsWith(".")) continue;

      const modelPath = path.join(orgPath, subEntry.name);
      const configPath = path.join(modelPath, "config.json");

      if (fs.existsSync(configPath)) {
        try {
          const config: ModelConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
          models.push({
            id: `${entry.name}/${subEntry.name}`,
            name: config.model_name || `${entry.name}/${subEntry.name}`,
            path: modelPath,
            config,
          });
        } catch (e) {
          console.error(`Failed to parse config for ${entry.name}/${subEntry.name}:`, e);
        }
      }
    }
  }

  return models;
}

export function loadChatPairs(modelPath: string): ConversationPair[] {
  const chatPairsPath = path.join(modelPath, "chat_pairs.json");
  if (!fs.existsSync(chatPairsPath)) {
    return [];
  }
  return JSON.parse(fs.readFileSync(chatPairsPath, "utf-8"));
}

export function loadTrainingPairs(modelPath: string): ConversationPair[] {
  const trainingPairsPath = path.join(modelPath, "training_pairs.json");
  if (!fs.existsSync(trainingPairsPath)) {
    return [];
  }
  return JSON.parse(fs.readFileSync(trainingPairsPath, "utf-8"));
}

export function loadBasePredictions(modelPath: string): PredictionRow[] {
  const csvPath = path.join(modelPath, "eval_bleurt_bertscore_per_example.csv");
  if (!fs.existsSync(csvPath)) {
    return [];
  }
  const content = fs.readFileSync(csvPath, "utf-8");
  const result = Papa.parse<PredictionRow>(content, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data;
}

export function loadFineTunedPredictions(modelPath: string): PredictionRow[] {
  const csvPath = path.join(modelPath, "eval_ft_bleurt_bertscore_per_example.csv");
  if (!fs.existsSync(csvPath)) {
    return [];
  }
  const content = fs.readFileSync(csvPath, "utf-8");
  const result = Papa.parse<PredictionRow>(content, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data;
}

export function getTurnLengthBucket(numTurns: number, buckets: SamplingConfig["turnLengthBuckets"]): "short" | "medium" | "long" {
  if (numTurns >= buckets.short[0] && numTurns <= buckets.short[1]) return "short";
  if (numTurns >= buckets.medium[0] && numTurns <= buckets.medium[1]) return "medium";
  return "long";
}

export function createStratifiedSample(samples: { index: number; dataset: string; numTurns: number }[], config: SamplingConfig): number[] {
  const random = seededRandom(config.seed);
  const targetCount = Math.ceil(samples.length * (config.samplePercentage / 100));

  // Group by stratification keys
  const groups: Record<string, number[]> = {};

  for (const sample of samples) {
    let key = "";
    if (config.stratifyBy.includes("dataset")) {
      key += sample.dataset;
    }
    if (config.stratifyBy.includes("turnLength")) {
      key += `_${getTurnLengthBucket(sample.numTurns, config.turnLengthBuckets)}`;
    }
    if (!key) key = "all";

    if (!groups[key]) groups[key] = [];
    groups[key].push(sample.index);
  }

  // Sample proportionally from each group
  const selectedIndices: number[] = [];
  const groupKeys = Object.keys(groups);
  const samplesPerGroup = Math.ceil(targetCount / groupKeys.length);

  for (const key of groupKeys) {
    const groupIndices = shuffleArray(groups[key], random);
    const toTake = Math.min(samplesPerGroup, groupIndices.length);
    selectedIndices.push(...groupIndices.slice(0, toTake));
  }

  // Shuffle final selection
  return shuffleArray(selectedIndices, random).slice(0, targetCount);
}

interface MergedCondition {
  pred: string | null;
  metrics: {
    bertscore_f1: number | null;
    bleurt: number | null;
    ppl: number | null;
  };
}

interface MergedPrediction {
  index: number;
  ground_truth: string;
  dataset: string;
  num_turns: number;
  conditions: Record<string, MergedCondition>;
}

function loadMergedPredictions(): Record<string, MergedPrediction[]> | null {
  const mergedPath = path.join(process.cwd(), "data", "merged_predictions.json");
  if (!fs.existsSync(mergedPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(mergedPath, "utf-8"));
}

function displayNameForCondition(conditionId: string): string {
  return conditionId
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function buildEvaluationSamples(models: ModelData[], samplingConfig: SamplingConfig = DEFAULT_SAMPLING_CONFIG): EvaluationSample[] {
  if (models.length === 0) return [];

  // Try to load pre-merged predictions (joined by ground truth in Python)
  const mergedData = loadMergedPredictions();

  // Use first model as reference for chat pairs
  const referenceModel = models[0];
  const evalPairs = loadChatPairs(referenceModel.path);

  // Create lookup maps for merged predictions by ground truth
  const predictionsByGroundTruth: Record<string, Record<string, MergedPrediction>> = {};
  if (mergedData) {
    for (const [modelId, predictions] of Object.entries(mergedData)) {
      predictionsByGroundTruth[modelId] = {};
      for (const pred of predictions) {
        predictionsByGroundTruth[modelId][pred.ground_truth.trim()] = pred;
      }
    }
  }

  // Create eval pairs with indices
  const allPairs = evalPairs.map((p, i) => ({ ...p, index: i, split: "eval" as const }));

  // Create sampling metadata
  const samplingMeta = allPairs.map((p, i) => ({
    index: i,
    dataset: p.meta.dataset.includes("WildChat") ? "wildchat" : "sgd",
    numTurns: p.meta.num_turns || p.conversation.length,
  }));

  // Get stratified sample indices
  const selectedIndices = createStratifiedSample(samplingMeta, samplingConfig);

  // Build evaluation samples
  const samples: EvaluationSample[] = [];
  const random = seededRandom(samplingConfig.seed + 1); // Different seed for blinding

  for (const idx of selectedIndices) {
    const pair = allPairs[idx];
    if (!pair) continue;

    const groundTruth = pair.target_user.trim();

    const conditionSet = new Set<string>();
    for (const model of models) {
      const pred = predictionsByGroundTruth[model.id]?.[groundTruth];
      Object.keys(pred?.conditions || {}).forEach((conditionId) => conditionSet.add(`${model.id}:${conditionId}`));
    }
    const predictionEntries = Array.from(conditionSet);
    const shuffledPreds = shuffleArray(predictionEntries, random);
    const blindedMapping: Record<string, string> = {};
    const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    shuffledPreds.forEach((entry, i) => {
      blindedMapping[labels[i]] = entry;
    });

    const predictions: PredictionCondition[] = [];
    for (const model of models) {
      const modelPreds = predictionsByGroundTruth[model.id];
      const pred = modelPreds?.[groundTruth] || null;
      for (const [conditionId, condition] of Object.entries(pred?.conditions || {})) {
        predictions.push({
          conditionId: `${model.id}:${conditionId}`,
          displayName: `${model.name} · ${displayNameForCondition(conditionId)}`,
          text: condition.pred || "",
          metrics: {
            bertscore_f1: condition.metrics?.bertscore_f1,
            bleurt: condition.metrics?.bleurt,
            ppl_content: condition.metrics?.ppl,
          } satisfies ConditionMetrics,
        });
      }
    }

    samples.push({
      id: `${pair.split}-${pair.index}`,
      sampleIndex: idx,
      dataset: pair.meta.dataset,
      context: pair.conversation,
      groundTruth: pair.target_user,
      predictions,
      blindedMapping,
    });
  }

  return samples;
}

export function saveRatingsToCSV(modelPath: string, ratings: { sampleId: string; ratings: Record<string, number | string> }[]): void {
  const outputPath = path.join(modelPath, "human_eval_ratings.csv");
  const csv = Papa.unparse(ratings);
  fs.writeFileSync(outputPath, csv);
}

export function loadExistingRatings(modelPath: string): Record<string, unknown>[] {
  const ratingsPath = path.join(modelPath, "human_eval_ratings.csv");
  if (!fs.existsSync(ratingsPath)) {
    return [];
  }
  const content = fs.readFileSync(ratingsPath, "utf-8");
  const result = Papa.parse(content, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return result.data as Record<string, unknown>[];
}

export function getCorpusStats(models: ModelData[]): {
  totalSamples: number;
  evalSamples: number;
  trainingSamples: number;
  datasetBreakdown: Record<string, number>;
} {
  if (models.length === 0) {
    return {
      totalSamples: 0,
      evalSamples: 0,
      trainingSamples: 0,
      datasetBreakdown: {},
    };
  }

  const referenceModel = models[0];
  const evalPairs = loadChatPairs(referenceModel.path);
  const trainingPairs = loadTrainingPairs(referenceModel.path);

  const datasetBreakdown: Record<string, number> = {};
  for (const pair of [...evalPairs, ...trainingPairs]) {
    const dataset = pair.meta.dataset.includes("WildChat") ? "WildChat" : "SGD";
    datasetBreakdown[dataset] = (datasetBreakdown[dataset] || 0) + 1;
  }

  return {
    totalSamples: evalPairs.length + trainingPairs.length,
    evalSamples: evalPairs.length,
    trainingSamples: trainingPairs.length,
    datasetBreakdown,
  };
}
