import { NextRequest, NextResponse } from "next/server";
import {
  loadSession,
  addRatingToSession,
  addLabelValidationToSession,
  exportSessionToCSV,
  exportRatingsToModelFolders,
  getSessionStats,
} from "@/lib/session-store";
import { Rating, LabelValidation } from "@/types/evaluation";

export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const session = loadSession(id);

    if (!session) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 });
    }

    const stats = getSessionStats(id);

    return NextResponse.json({ session, stats });
  } catch (error) {
    console.error("Error loading session:", error);
    return NextResponse.json({ error: "Failed to load session" }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { action, rating, validation } = body;

    if (action === "addRating" && rating) {
      const ratingData: Rating = {
        ...rating,
        timestamp: new Date().toISOString(),
      };
      const session = addRatingToSession(id, ratingData);
      if (!session) {
        return NextResponse.json({ error: "Session not found" }, { status: 404 });
      }
      return NextResponse.json({ session });
    }

    if (action === "addValidation" && validation) {
      const validationData: LabelValidation = {
        ...validation,
        timestamp: new Date().toISOString(),
      };
      const session = addLabelValidationToSession(id, validationData);
      if (!session) {
        return NextResponse.json({ error: "Session not found" }, { status: 404 });
      }
      return NextResponse.json({ session });
    }

    if (action === "export") {
      const { ratingsCSV, validationsCSV } = exportSessionToCSV(id);
      return NextResponse.json({ ratingsCSV, validationsCSV });
    }

    if (action === "exportToModels") {
      exportRatingsToModelFolders(id);
      return NextResponse.json({ success: true });
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Error updating session:", error);
    return NextResponse.json({ error: "Failed to update session" }, { status: 500 });
  }
}
