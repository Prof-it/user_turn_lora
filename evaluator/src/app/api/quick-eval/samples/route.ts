import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

interface Message {
  role: string;
  content: string;
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
    type: "baseline" | "finetuned" | "prompt_zeroshot" | "prompt_fewshot";
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

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const sampleCount = parseInt(searchParams.get("count") || "250");
    const seed = parseInt(searchParams.get("seed") || "42");

    const rootDir = path.join(process.cwd(), "..");

    // Load model predictions (pick one model - Qwen as representative)
    const modelDir = path.join(rootDir, "outputs", "Qwen-Qwen2.5-3B-Instruct");
    const chatPairsPath = path.join(modelDir, "chat_pairs.json");

    if (!fs.existsSync(chatPairsPath)) {
      return NextResponse.json({ error: "chat_pairs.json not found" }, { status: 404 });
    }

    const chatPairs = JSON.parse(fs.readFileSync(chatPairsPath, "utf-8"));

    // Load baseline and finetuned predictions from eval CSVs
    // CSV format: ref,pred,bertscore_f1,bleurt,ppl_content,dataset,...
    const evalBasePath = path.join(modelDir, "eval_bleurt_bertscore_per_example.csv");
    const evalFtPath = path.join(modelDir, "eval_ft_bleurt_bertscore_per_example.csv");

    // Proper CSV parser that handles multi-line quoted fields
    function parseCSV(content: string): string[][] {
      const rows: string[][] = [];
      let currentRow: string[] = [];
      let currentField = "";
      let inQuotes = false;

      for (let i = 0; i < content.length; i++) {
        const char = content[i];
        const nextChar = content[i + 1];

        if (inQuotes) {
          if (char === '"' && nextChar === '"') {
            currentField += '"';
            i++; // Skip escaped quote
          } else if (char === '"') {
            inQuotes = false;
          } else {
            currentField += char;
          }
        } else {
          if (char === '"') {
            inQuotes = true;
          } else if (char === ",") {
            currentRow.push(currentField);
            currentField = "";
          } else if (char === "\n" || (char === "\r" && nextChar === "\n")) {
            currentRow.push(currentField);
            if (currentRow.length > 1) rows.push(currentRow);
            currentRow = [];
            currentField = "";
            if (char === "\r") i++; // Skip \r\n
          } else if (char !== "\r") {
            currentField += char;
          }
        }
      }
      // Handle last row
      if (currentField || currentRow.length > 0) {
        currentRow.push(currentField);
        if (currentRow.length > 1) rows.push(currentRow);
      }
      return rows;
    }

    // Build lookup maps by ground truth (ref) for proper alignment
    const baselineByRef: Record<string, string> = {};
    const finetunedByRef: Record<string, string> = {};

    if (fs.existsSync(evalBasePath)) {
      const baseCSV = fs.readFileSync(evalBasePath, "utf-8");
      const rows = parseCSV(baseCSV).slice(1); // Skip header
      // CSV columns: ref, pred, bertscore_f1, bleurt, ...
      rows.forEach((row) => {
        const ref = row[0] || "";
        const pred = row[1] || "";
        if (ref) baselineByRef[ref] = pred;
      });
    }

    if (fs.existsSync(evalFtPath)) {
      const ftCSV = fs.readFileSync(evalFtPath, "utf-8");
      const rows = parseCSV(ftCSV).slice(1);
      rows.forEach((row) => {
        const ref = row[0] || "";
        const pred = row[1] || "";
        if (ref) finetunedByRef[ref] = pred;
      });
    }

    // Load prompt baseline predictions (zero-shot and few-shot)
    const zeroShotPath = path.join(rootDir, "outputs", "prompt_baseline", "gpt-4o-mini-zero-shot", "predictions.json");
    const fewShotPath = path.join(rootDir, "outputs", "prompt_baseline", "gpt-4o-mini-few-shot", "predictions.json");
    let zeroShotPreds: string[] = [];
    let fewShotPreds: string[] = [];

    if (fs.existsSync(zeroShotPath)) {
      const data = JSON.parse(fs.readFileSync(zeroShotPath, "utf-8"));
      zeroShotPreds = data.map((item: { pred_prompt_baseline?: string }) => item.pred_prompt_baseline || "");
    }

    if (fs.existsSync(fewShotPath)) {
      const data = JSON.parse(fs.readFileSync(fewShotPath, "utf-8"));
      fewShotPreds = data.map((item: { pred_prompt_baseline?: string }) => item.pred_prompt_baseline || "");
    }

    // Build samples with all 4 conditions, aligned by ground truth
    const allSamples: Sample[] = [];

    for (let i = 0; i < chatPairs.length; i++) {
      const pair = chatPairs[i];
      const groundTruth = pair.target_user;

      // Look up predictions by ground truth text for proper alignment
      const basePred = baselineByRef[groundTruth] || "";
      const ftPred = finetunedByRef[groundTruth] || "";
      const zsPred = zeroShotPreds[i] || ""; // Prompt baselines are already aligned with chat_pairs
      const fsPred = fewShotPreds[i] || "";

      // Skip if any prediction is missing
      if (!basePred || !ftPred || !zsPred || !fsPred) continue;

      // Create blinded labels (A, B, C, D) with randomized order per sample
      const conditions: { label: string; text: string; type: "baseline" | "finetuned" | "prompt_zeroshot" | "prompt_fewshot" }[] = [
        { label: "", text: basePred, type: "baseline" },
        { label: "", text: ftPred, type: "finetuned" },
        { label: "", text: zsPred, type: "prompt_zeroshot" },
        { label: "", text: fsPred, type: "prompt_fewshot" },
      ];

      // Shuffle conditions for this sample
      const shuffled = shuffleArray(conditions, seed + i);
      const labels = ["A", "B", "C", "D"];
      shuffled.forEach((cond, idx) => {
        cond.label = labels[idx];
      });

      // Build blinded mapping (revealed after rating)
      const blindedMapping: Record<string, string> = {};
      shuffled.forEach((cond) => {
        blindedMapping[cond.label] = cond.type;
      });

      allSamples.push({
        id: `sample_${i}`,
        index: i,
        dataset: pair.meta?.dataset || "unknown",
        context: pair.conversation,
        groundTruth: pair.target_user,
        predictions: shuffled,
        blindedMapping,
      });
    }

    // Shuffle and limit samples
    const shuffledSamples = shuffleArray(allSamples, seed);
    const selectedSamples = shuffledSamples.slice(0, sampleCount);

    return NextResponse.json({
      samples: selectedSamples,
      totalAvailable: allSamples.length,
      seed,
    });
  } catch (error) {
    console.error("Error loading samples:", error);
    return NextResponse.json({ error: "Failed to load samples" }, { status: 500 });
  }
}
