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

    return NextResponse.json({
      samples,
      config: samplingConfig,
      totalSamples: samples.length,
    });
  } catch (error) {
    console.error("Error building samples:", error);
    return NextResponse.json({ error: "Failed to build evaluation samples" }, { status: 500 });
  }
}
