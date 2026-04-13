import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

import type { Prediction, PredictionRating } from "@/lib/quick-eval";
import type { RatingCategory } from "@/hooks/use-rating-keyboard";


interface PredictionCardProps {
  color: { border: string; bg: string; text: string; circle: string };
  prediction: Prediction;
  scores: PredictionRating;
  activeCategory?: RatingCategory | null;
  onSelectCategory?: (category: RatingCategory) => void;
  onUpdate: (category: keyof PredictionRating, value: number) => void;
}

function averageScore(score: PredictionRating): number {
  return (score.relevance + score.coherence + score.naturalness) / 3;
}

export function PredictionCard({ color, prediction, scores, activeCategory, onSelectCategory, onUpdate }: PredictionCardProps) {
  return (
    <Card className={`border-2 ${color.border}`}>
      <CardHeader className={`pb-2 ${color.bg}`}>
        <CardTitle className="text-lg flex items-center gap-2">
          <span className={`w-8 h-8 rounded-full ${color.circle} text-white flex items-center justify-center font-bold`}>
            {prediction.label}
          </span>
          Prediction {prediction.label}
        </CardTitle>
        <p className="text-xs text-gray-500">{prediction.type}</p>
      </CardHeader>
      <CardContent className="pt-3 space-y-4">
        <div className="bg-white p-3 rounded border max-h-[150px] overflow-y-auto">
          <p className="text-sm whitespace-pre-wrap">{prediction.text}</p>
        </div>

        {(["relevance", "coherence", "naturalness"] as const).map((category) => {
          const isActive = activeCategory === category;
          return (
          <div
            key={category}
            role="button"
            tabIndex={0}
            aria-label={`Select ${prediction.label} ${category}`}
            onClick={() => onSelectCategory?.(category)}
            onFocus={() => onSelectCategory?.(category)}
            className={`space-y-2 rounded-md p-2 -m-2 outline-none ${
              isActive ? "ring-2 ring-blue-500 ring-offset-2 bg-blue-50" : ""
            }`}
          >
            <div className="flex justify-between items-center">
              <Label className="text-sm font-medium capitalize">{category}</Label>
              <span className={`text-xl font-bold ${color.text}`}>{scores[category]}</span>
            </div>
            <Slider
              value={[scores[category]]}
              onValueChange={(value) => onUpdate(category, value[0])}
              min={1}
              max={5}
              step={1}
              className="h-3"
            />
          </div>
          );
        })}

        <div className="text-center text-sm text-gray-500 pt-2 border-t">
          Avg: <span className="font-bold">{averageScore(scores).toFixed(1)}</span>
        </div>
      </CardContent>
    </Card>
  );
}
