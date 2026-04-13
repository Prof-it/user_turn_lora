import { useEffect, type Dispatch, type SetStateAction } from "react";

import type { PredictionRating, Sample } from "@/lib/quick-eval";

export const RATING_CATEGORIES = ["relevance", "coherence", "naturalness"] as const;

export type RatingCategory = (typeof RATING_CATEGORIES)[number];

export interface RatingCursor {
  predictionIndex: number;
  categoryIndex: number;
}

interface RatingKeyboardOptions {
  sample: Sample | undefined;
  scores: Record<string, PredictionRating>;
  cursor: RatingCursor;
  setCursor: Dispatch<SetStateAction<RatingCursor>>;
  onUpdate: (label: string, category: keyof PredictionRating, value: number) => void;
  onSubmit: () => void;
  onPrevious: () => void;
  onSkip: () => void;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  return target.isContentEditable || tagName === "input" || tagName === "textarea" || tagName === "select";
}

function clampScore(value: number): number {
  return Math.max(1, Math.min(5, value));
}

function movedCursor(cursor: RatingCursor, delta: number, predictionCount: number): RatingCursor {
  const categoryCount = RATING_CATEGORIES.length;
  const totalFields = Math.max(predictionCount * categoryCount, 1);
  const flatIndex = cursor.predictionIndex * categoryCount + cursor.categoryIndex;
  const nextIndex = (flatIndex + delta + totalFields) % totalFields;
  return {
    predictionIndex: Math.floor(nextIndex / categoryCount),
    categoryIndex: nextIndex % categoryCount,
  };
}

export function useRatingKeyboard({
  sample,
  scores,
  cursor,
  setCursor,
  onUpdate,
  onSubmit,
  onPrevious,
  onSkip,
}: RatingKeyboardOptions) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (!sample || isEditableTarget(event.target)) {
        return;
      }

      const prediction = sample.predictions[cursor.predictionIndex];
      const category = RATING_CATEGORIES[cursor.categoryIndex];
      const score = prediction ? scores[prediction.label]?.[category] ?? 3 : 3;
      const key = event.key.toLowerCase();

      if (event.key === "ArrowDown") {
        event.preventDefault();
        setCursor((current) => movedCursor(current, 1, sample.predictions.length));
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        setCursor((current) => movedCursor(current, -1, sample.predictions.length));
        return;
      }
      if (event.key === "ArrowRight" && prediction) {
        event.preventDefault();
        onUpdate(prediction.label, category, clampScore(score + 1));
        return;
      }
      if (event.key === "ArrowLeft" && prediction) {
        event.preventDefault();
        onUpdate(prediction.label, category, clampScore(score - 1));
        return;
      }
      if (/^[1-5]$/.test(event.key) && prediction) {
        event.preventDefault();
        onUpdate(prediction.label, category, Number(event.key));
        return;
      }
      if (key === "enter") {
        event.preventDefault();
        onSubmit();
        return;
      }
      if (key === "p") {
        event.preventDefault();
        onPrevious();
        return;
      }
      if (key === "s") {
        event.preventDefault();
        onSkip();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [cursor, onPrevious, onSkip, onSubmit, onUpdate, sample, scores, setCursor]);
}
