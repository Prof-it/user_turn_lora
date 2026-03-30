import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

interface Message {
  role: string;
  content: string;
}

interface MergedCondition {
  pred: string | null;
}

interface MergedSample {
  index: number;
  ground_truth: string;
  dataset: string;
  num_turns: number;
  conversation_hash: string;
  conditions: Record<string, MergedCondition>;
}

interface Sample {
  id: string;
  index: number;
  dataset: string;
  context: Message[];
  groundTruth: string;
  predictions: {
    label: string;
    text: string;
    type: string;
  }[];
  blindedMapping: Record<string, string>;
}

function shuffleArray<T>(array: T[], seed: number): T[] {
  const result = [...array];
  let currentIndex = result.length;
  const seededRandom = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  while (currentIndex !== 0) {
    const randomIndex = Math.floor(seededRandom() * currentIndex);
    currentIndex--;
    [result[currentIndex], result[randomIndex]] = [result[randomIndex], result[currentIndex]];
  }

  return result;
}

function loadJson<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function buildExampleKey(conversationHash: string, groundTruth: string, dataset: string): string {
  return `${dataset}::${conversationHash}::${groundTruth.trim()}`;
}

function loadOpenAIPredictions(rootDir: string, mode: "zero-shot" | "few-shot"): Record<string, string> {
  const pathName = path.join(rootDir, "outputs", "prompt_baseline", `gpt-4o-mini-${mode}`, "predictions.json");
  const rows =
    loadJson<
      Array<{
        target_user: string;
        pred_prompt_baseline?: string;
        meta?: { conversation_hash?: string; dataset?: string };
      }>
    >(pathName) || [];
  return Object.fromEntries(
    rows.map((row) => [
      buildExampleKey(row.meta?.conversation_hash || "", row.target_user, row.meta?.dataset || ""),
      row.pred_prompt_baseline || "",
    ]),
  );
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const sampleCount = parseInt(searchParams.get("count") || "250");
    const seed = parseInt(searchParams.get("seed") || "42");
    const includeOpenAI = searchParams.get("includeOpenAI") === "1";
    const modelId = searchParams.get("model") || "outputs/Qwen-Qwen2.5-3B-Instruct";

    const rootDir = path.join(process.cwd(), "..");
    const mergedPath = path.join(process.cwd(), "data", "merged_predictions.json");
    const merged = loadJson<Record<string, MergedSample[]>>(mergedPath);
    if (!merged || !merged[modelId]) {
      return NextResponse.json({ error: `Merged predictions not found for ${modelId}` }, { status: 404 });
    }

    const modelDir = path.join(rootDir, modelId);
    const chatPairs = loadJson<Array<{ conversation: Message[]; target_user: string; meta?: { dataset?: string } }>>(
      path.join(modelDir, "chat_pairs.json"),
    );
    if (!chatPairs) {
      return NextResponse.json({ error: "chat_pairs.json not found" }, { status: 404 });
    }

    const openAIZeroShot = includeOpenAI ? loadOpenAIPredictions(rootDir, "zero-shot") : {};
    const openAIFewShot = includeOpenAI ? loadOpenAIPredictions(rootDir, "few-shot") : {};
    const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    const allSamples: Sample[] = [];
    for (const pair of merged[modelId]) {
      const chatPair = chatPairs[pair.index];
      if (!chatPair) continue;

      const conditions = Object.entries(pair.conditions)
        .filter(([, condition]) => Boolean(condition.pred))
        .map(([conditionId, condition]) => ({
          label: "",
          text: condition.pred || "",
          type: conditionId,
        }));

      const exampleKey = buildExampleKey(pair.conversation_hash, pair.ground_truth, pair.dataset);
      if (includeOpenAI) {
        if (openAIZeroShot[exampleKey]) {
          conditions.push({ label: "", text: openAIZeroShot[exampleKey], type: "openai_zero_shot" });
        }
        if (openAIFewShot[exampleKey]) {
          conditions.push({ label: "", text: openAIFewShot[exampleKey], type: "openai_few_shot" });
        }
      }

      if (conditions.length < 2) {
        continue;
      }

      const shuffled = shuffleArray(conditions, seed + pair.index);
      shuffled.forEach((condition, idx) => {
        condition.label = labels[idx];
      });

      const blindedMapping: Record<string, string> = {};
      for (const condition of shuffled) {
        blindedMapping[condition.label] = condition.type;
      }

      allSamples.push({
        id: `sample_${pair.index}`,
        index: pair.index,
        dataset: pair.dataset,
        context: chatPair.conversation,
        groundTruth: chatPair.target_user,
        predictions: shuffled,
        blindedMapping,
      });
    }

    const shuffledSamples = shuffleArray(allSamples, seed);
    const selectedSamples = shuffledSamples.slice(0, sampleCount);

    return NextResponse.json({
      samples: selectedSamples,
      totalAvailable: allSamples.length,
      seed,
      modelId,
      includeOpenAI,
    });
  } catch (error) {
    console.error("Error loading samples:", error);
    return NextResponse.json({ error: "Failed to load samples" }, { status: 500 });
  }
}
