import { NextRequest, NextResponse } from "next/server";
import { discoverModels, buildEvaluationSamples } from "@/lib/data-loader";
import { SamplingConfig, DEFAULT_SAMPLING_CONFIG } from "@/types/evaluation";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const modelIds = searchParams.get("models")?.split(",") || [];
    const seed = parseInt(searchParams.get("seed") || String(DEFAULT_SAMPLING_CONFIG.seed));
    const samplePercentage = parseFloat(searchParams.get("samplePercentage") || String(DEFAULT_SAMPLING_CONFIG.samplePercentage));
    const includeTraining = searchParams.get("includeTraining") !== "false";

    const allModels = discoverModels();
    const selectedModels = modelIds.length > 0 ? allModels.filter((m) => modelIds.includes(m.id)) : allModels;

    if (selectedModels.length === 0) {
      return NextResponse.json({ error: "No models found" }, { status: 404 });
    }

    const samplingConfig: SamplingConfig = {
      ...DEFAULT_SAMPLING_CONFIG,
      seed,
      samplePercentage,
    };

    const samples = buildEvaluationSamples(selectedModels, samplingConfig, includeTraining);

    // Create reproducible sample_ids export
    const sampleIds = {
      generated_at: new Date().toISOString(),
      config: {
        seed: samplingConfig.seed,
        samplePercentage: samplingConfig.samplePercentage,
        stratifyBy: samplingConfig.stratifyBy,
        includeTraining,
        models: selectedModels.map((m) => m.id),
      },
      total_samples: samples.length,
      samples: samples.map((s) => ({
        id: s.id,
        dataset: s.dataset,
        sampleIndex: s.sampleIndex,
      })),
    };

    return NextResponse.json(sampleIds);
  } catch (error) {
    console.error("Error exporting sample IDs:", error);
    return NextResponse.json({ error: "Failed to export sample IDs" }, { status: 500 });
  }
}
