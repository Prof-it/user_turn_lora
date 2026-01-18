import { NextResponse } from "next/server";
import { discoverModels, getCorpusStats } from "@/lib/data-loader";

export async function GET() {
  try {
    const models = discoverModels();
    const stats = getCorpusStats(models);

    return NextResponse.json({
      models,
      stats,
    });
  } catch (error) {
    console.error("Error discovering models:", error);
    return NextResponse.json({ error: "Failed to discover models" }, { status: 500 });
  }
}
