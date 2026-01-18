import fs from "fs";
import path from "path";
import Papa from "papaparse";
import { EvaluationSession, Rating, LabelValidation } from "@/types/evaluation";

const ROOT_DIR = path.resolve(process.cwd(), "..");
const SESSIONS_DIR = path.join(process.cwd(), "data", "sessions");

// Ensure sessions directory exists
if (!fs.existsSync(SESSIONS_DIR)) {
  fs.mkdirSync(SESSIONS_DIR, { recursive: true });
}

export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export function saveSession(session: EvaluationSession): void {
  const sessionPath = path.join(SESSIONS_DIR, `${session.id}.json`);
  fs.writeFileSync(sessionPath, JSON.stringify(session, null, 2));
}

export function loadSession(sessionId: string): EvaluationSession | null {
  const sessionPath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  if (!fs.existsSync(sessionPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(sessionPath, "utf-8"));
}

export function listSessions(): { id: string; raterId: string; startedAt: string; progress: { completed: number; total: number } }[] {
  if (!fs.existsSync(SESSIONS_DIR)) {
    return [];
  }
  const files = fs.readdirSync(SESSIONS_DIR).filter((f) => f.endsWith(".json"));
  return files.map((f) => {
    const session = JSON.parse(fs.readFileSync(path.join(SESSIONS_DIR, f), "utf-8")) as EvaluationSession;
    return {
      id: session.id,
      raterId: session.raterId,
      startedAt: session.startedAt,
      progress: session.progress,
    };
  });
}

export function addRatingToSession(sessionId: string, rating: Rating): EvaluationSession | null {
  const session = loadSession(sessionId);
  if (!session) return null;

  // Check if rating for this sample already exists
  const existingIndex = session.ratings.findIndex((r) => r.sampleId === rating.sampleId && r.blindedLabel === rating.blindedLabel);

  if (existingIndex >= 0) {
    session.ratings[existingIndex] = rating;
  } else {
    session.ratings.push(rating);
  }

  // Update progress
  const uniqueSamplesRated = new Set(session.ratings.map((r) => r.sampleId)).size;
  session.progress.completed = uniqueSamplesRated;

  saveSession(session);
  return session;
}

export function addLabelValidationToSession(sessionId: string, validation: LabelValidation): EvaluationSession | null {
  const session = loadSession(sessionId);
  if (!session) return null;

  const existingIndex = session.labelValidations.findIndex((v) => v.sampleId === validation.sampleId);

  if (existingIndex >= 0) {
    session.labelValidations[existingIndex] = validation;
  } else {
    session.labelValidations.push(validation);
  }

  saveSession(session);
  return session;
}

export function exportSessionToCSV(sessionId: string): {
  ratingsCSV: string;
  validationsCSV: string;
} {
  const session = loadSession(sessionId);
  if (!session) {
    return { ratingsCSV: "", validationsCSV: "" };
  }

  const ratingsCSV = Papa.unparse(session.ratings);
  const validationsCSV = Papa.unparse(session.labelValidations);

  return { ratingsCSV, validationsCSV };
}

export function exportRatingsToModelFolders(sessionId: string): void {
  const session = loadSession(sessionId);
  if (!session) return;

  // Group ratings by model
  const ratingsByModel: Record<string, Rating[]> = {};

  for (const rating of session.ratings) {
    const modelId = rating.modelId;
    if (!ratingsByModel[modelId]) {
      ratingsByModel[modelId] = [];
    }
    ratingsByModel[modelId].push(rating);
  }

  // Export to each model folder
  for (const [modelId, ratings] of Object.entries(ratingsByModel)) {
    const modelPath = path.join(ROOT_DIR, modelId);
    if (!fs.existsSync(modelPath)) continue;

    const outputPath = path.join(modelPath, "human_eval_ratings.csv");
    const csv = Papa.unparse(ratings);
    fs.writeFileSync(outputPath, csv);
  }
}

export function getSessionStats(sessionId: string): {
  totalRatings: number;
  ratingsByModel: Record<string, number>;
  ratingsByCategory: Record<string, { avg: number; count: number }>;
  labelValidations: { valid: number; invalid: number };
} | null {
  const session = loadSession(sessionId);
  if (!session) return null;

  const ratingsByModel: Record<string, number> = {};
  const categoryScores: Record<string, number[]> = {
    relevance: [],
    coherence: [],
    naturalness: [],
    specificity: [],
  };

  for (const rating of session.ratings) {
    ratingsByModel[rating.modelId] = (ratingsByModel[rating.modelId] || 0) + 1;
    categoryScores.relevance.push(rating.relevance);
    categoryScores.coherence.push(rating.coherence);
    categoryScores.naturalness.push(rating.naturalness);
    if (rating.specificity) {
      categoryScores.specificity.push(rating.specificity);
    }
  }

  const ratingsByCategory: Record<string, { avg: number; count: number }> = {};
  for (const [cat, scores] of Object.entries(categoryScores)) {
    if (scores.length > 0) {
      ratingsByCategory[cat] = {
        avg: scores.reduce((a, b) => a + b, 0) / scores.length,
        count: scores.length,
      };
    }
  }

  const validCount = session.labelValidations.filter((v) => v.isValid).length;
  const invalidCount = session.labelValidations.filter((v) => !v.isValid).length;

  return {
    totalRatings: session.ratings.length,
    ratingsByModel,
    ratingsByCategory,
    labelValidations: { valid: validCount, invalid: invalidCount },
  };
}
