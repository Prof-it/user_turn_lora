export interface Prediction {
  label: string;
  text: string;
  type: string;
}

export interface Sample {
  id: string;
  index: number;
  dataset: string;
  context: { role: string; content: string }[];
  groundTruth: string;
  predictions: Prediction[];
  blindedMapping: Record<string, string>;
}

export interface PredictionRating {
  relevance: number;
  coherence: number;
  naturalness: number;
}

export interface Rating {
  sampleId: string;
  scores: Record<string, PredictionRating>;
  timestamp: string;
}

export function createDefaultScores(predictions: Prediction[]): Record<string, PredictionRating> {
  return Object.fromEntries(
    predictions.map((prediction) => [
      prediction.label,
      { relevance: 3, coherence: 3, naturalness: 3 } satisfies PredictionRating,
    ]),
  );
}

function averageScore(score: PredictionRating): number {
  return (score.relevance + score.coherence + score.naturalness) / 3;
}

export function buildExportCsv(samples: Sample[], ratings: Rating[]): string {
  const conditionIds = Array.from(
    new Set(samples.flatMap((sample) => Object.values(sample.blindedMapping))),
  ).sort();

  const headers = ["sample_id", "dataset"];
  for (const conditionId of conditionIds) {
    headers.push(
      `${conditionId}__relevance`,
      `${conditionId}__coherence`,
      `${conditionId}__naturalness`,
      `${conditionId}__avg`,
    );
  }
  headers.push("winner", "timestamp");

  const rows = ratings
    .map((rating) => {
      const sample = samples.find((entry) => entry.id === rating.sampleId);
      if (!sample) return null;

      const scoresByCondition: Record<string, PredictionRating> = {};
      for (const [label, conditionId] of Object.entries(sample.blindedMapping)) {
        if (rating.scores[label]) {
          scoresByCondition[conditionId] = rating.scores[label];
        }
      }

      const row = [rating.sampleId, sample.dataset];
      let bestCondition = "";
      let bestScore = Number.NEGATIVE_INFINITY;
      let tie = false;

      for (const conditionId of conditionIds) {
        const score = scoresByCondition[conditionId] || { relevance: 0, coherence: 0, naturalness: 0 };
        const avg = averageScore(score);
        row.push(
          String(score.relevance),
          String(score.coherence),
          String(score.naturalness),
          avg.toFixed(2),
        );
        if (avg > bestScore) {
          bestScore = avg;
          bestCondition = conditionId;
          tie = false;
        } else if (avg === bestScore) {
          tie = true;
        }
      }

      row.push(tie ? "tie" : bestCondition, rating.timestamp);
      return row.join(",");
    })
    .filter(Boolean);

  return [headers.join(","), ...rows].join("\n");
}
