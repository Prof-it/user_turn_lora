import { NextRequest, NextResponse } from "next/server";
import { generateSessionId, saveSession, listSessions } from "@/lib/session-store";
import { EvaluationSession } from "@/types/evaluation";

export async function GET() {
  try {
    const sessions = listSessions();
    return NextResponse.json({ sessions });
  } catch (error) {
    console.error("Error listing sessions:", error);
    return NextResponse.json({ error: "Failed to list sessions" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { raterId, selectedModels, sampleIds, seed } = body;

    if (!raterId) {
      return NextResponse.json({ error: "raterId is required" }, { status: 400 });
    }

    const session: EvaluationSession = {
      id: generateSessionId(),
      raterId,
      selectedModels: selectedModels || [],
      sampleIds: sampleIds || [],
      seed: seed || 42,
      startedAt: new Date().toISOString(),
      ratings: [],
      labelValidations: [],
      progress: {
        completed: 0,
        total: sampleIds?.length || 0,
      },
    };

    saveSession(session);

    return NextResponse.json({ session });
  } catch (error) {
    console.error("Error creating session:", error);
    return NextResponse.json({ error: "Failed to create session" }, { status: 500 });
  }
}
