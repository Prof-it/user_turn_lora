"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { ChevronRight, Save, Download, RotateCcw } from "lucide-react";

interface Prediction {
  label: string;
  text: string;
  type: "baseline" | "finetuned" | "prompt_zeroshot" | "prompt_fewshot";
}

interface Sample {
  id: string;
  index: number;
  dataset: string;
  context: { role: string; content: string }[];
  groundTruth: string;
  predictions: Prediction[];
  blindedMapping: Record<string, string>;
}

interface PredictionRating {
  relevance: number;
  coherence: number;
  naturalness: number;
}

interface Rating {
  sampleId: string;
  scores: Record<string, PredictionRating>;
  timestamp: string;
}

const STORAGE_KEY = "quick-eval-session-v3";
const AUTO_SAVE_INTERVAL = 5;

const createDefaultScores = (): Record<string, PredictionRating> => ({
  A: { relevance: 3, coherence: 3, naturalness: 3 },
  B: { relevance: 3, coherence: 3, naturalness: 3 },
  C: { relevance: 3, coherence: 3, naturalness: 3 },
  D: { relevance: 3, coherence: 3, naturalness: 3 },
});

export default function QuickEvalPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [scores, setScores] = useState<Record<string, PredictionRating>>(createDefaultScores());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetch("/api/quick-eval/samples?count=400");
        if (res.ok) {
          const data = await res.json();
          setSamples(data.samples);
        }

        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
          const session = JSON.parse(saved);
          setRatings(session.ratings || []);
          setCurrentIndex(session.currentIndex || 0);
          setLastSaved(session.lastSaved);
        }
      } catch (error) {
        console.error("Failed to load data:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const saveSession = useCallback(() => {
    const session = {
      ratings,
      currentIndex,
      lastSaved: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    setLastSaved(session.lastSaved);
    setSaving(false);
  }, [ratings, currentIndex]);

  useEffect(() => {
    if (ratings.length > 0 && ratings.length % AUTO_SAVE_INTERVAL === 0) {
      setSaving(true);
      saveSession();
    }
  }, [ratings.length, saveSession]);

  const currentSample = samples[currentIndex];

  const handleSubmitRating = () => {
    if (!currentSample) return;

    const newRating: Rating = {
      sampleId: currentSample.id,
      scores: JSON.parse(JSON.stringify(scores)),
      timestamp: new Date().toISOString(),
    };

    setRatings((prev) => [...prev.filter((r) => r.sampleId !== currentSample.id), newRating]);

    if (currentIndex < samples.length - 1) {
      setCurrentIndex((i) => i + 1);
      setScores(createDefaultScores());
    }
  };

  const handleExport = () => {
    const headers = [
      "sample_id",
      "dataset",
      "baseline_relevance",
      "baseline_coherence",
      "baseline_naturalness",
      "baseline_avg",
      "finetuned_relevance",
      "finetuned_coherence",
      "finetuned_naturalness",
      "finetuned_avg",
      "zeroshot_relevance",
      "zeroshot_coherence",
      "zeroshot_naturalness",
      "zeroshot_avg",
      "fewshot_relevance",
      "fewshot_coherence",
      "fewshot_naturalness",
      "fewshot_avg",
      "winner",
      "timestamp",
    ];

    const rows = ratings.map((r) => {
      const sample = samples.find((s) => s.id === r.sampleId);
      if (!sample) return null;

      const typeScores: Record<string, PredictionRating> = {
        baseline: { relevance: 0, coherence: 0, naturalness: 0 },
        finetuned: { relevance: 0, coherence: 0, naturalness: 0 },
        prompt_zeroshot: { relevance: 0, coherence: 0, naturalness: 0 },
        prompt_fewshot: { relevance: 0, coherence: 0, naturalness: 0 },
      };

      for (const [label, type] of Object.entries(sample.blindedMapping)) {
        if (r.scores[label]) {
          typeScores[type] = r.scores[label];
        }
      }

      const avg = (s: PredictionRating) => (s.relevance + s.coherence + s.naturalness) / 3;
      const baselineAvg = avg(typeScores.baseline);
      const finetunedAvg = avg(typeScores.finetuned);
      const zeroShotAvg = avg(typeScores.prompt_zeroshot);
      const fewShotAvg = avg(typeScores.prompt_fewshot);

      const maxAvg = Math.max(baselineAvg, finetunedAvg, zeroShotAvg, fewShotAvg);
      const winners = [];
      if (baselineAvg === maxAvg) winners.push("baseline");
      if (finetunedAvg === maxAvg) winners.push("finetuned");
      if (zeroShotAvg === maxAvg) winners.push("prompt_zeroshot");
      if (fewShotAvg === maxAvg) winners.push("prompt_fewshot");
      const winner = winners.length > 1 ? "tie" : winners[0];

      return [
        r.sampleId,
        sample.dataset,
        typeScores.baseline.relevance,
        typeScores.baseline.coherence,
        typeScores.baseline.naturalness,
        baselineAvg.toFixed(2),
        typeScores.finetuned.relevance,
        typeScores.finetuned.coherence,
        typeScores.finetuned.naturalness,
        finetunedAvg.toFixed(2),
        typeScores.prompt_zeroshot.relevance,
        typeScores.prompt_zeroshot.coherence,
        typeScores.prompt_zeroshot.naturalness,
        zeroShotAvg.toFixed(2),
        typeScores.prompt_fewshot.relevance,
        typeScores.prompt_fewshot.coherence,
        typeScores.prompt_fewshot.naturalness,
        fewShotAvg.toFixed(2),
        winner,
        r.timestamp,
      ].join(",");
    }).filter(Boolean);

    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `human_eval_ratings_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
  };

  const handleReset = () => {
    if (confirm("Are you sure you want to reset all ratings? This cannot be undone.")) {
      setRatings([]);
      setCurrentIndex(0);
      localStorage.removeItem(STORAGE_KEY);
      setLastSaved(null);
    }
  };

  const existingRating = currentSample
    ? ratings.find((r) => r.sampleId === currentSample.id)
    : null;

  useEffect(() => {
    if (existingRating) {
      setScores(JSON.parse(JSON.stringify(existingRating.scores)));
    } else {
      setScores(createDefaultScores());
    }
  }, [existingRating, currentIndex]);

  const updateScore = (label: string, category: keyof PredictionRating, value: number) => {
    setScores((prev) => ({
      ...prev,
      [label]: { ...prev[label], [category]: value },
    }));
  };

  const completedCount = ratings.length;
  const progress = samples.length > 0 ? Math.round((completedCount / samples.length) * 100) : 0;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading samples...</p>
        </div>
      </div>
    );
  }

  if (!currentSample) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="p-8 text-center">
            <h2 className="text-xl font-bold mb-4">Evaluation Complete!</h2>
            <p className="text-gray-600 mb-4">You have rated all {samples.length} samples.</p>
            <Button onClick={handleExport}>
              <Download className="h-4 w-4 mr-2" />
              Export Results
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const colors: Record<string, { border: string; bg: string; text: string; circle: string }> = {
    A: { border: "border-blue-200", bg: "bg-blue-50", text: "text-blue-600", circle: "bg-blue-600" },
    B: { border: "border-purple-200", bg: "bg-purple-50", text: "text-purple-600", circle: "bg-purple-600" },
    C: { border: "border-orange-200", bg: "bg-orange-50", text: "text-orange-600", circle: "bg-orange-600" },
    D: { border: "border-green-200", bg: "bg-green-50", text: "text-green-600", circle: "bg-green-600" },
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
          <div>
            <h1 className="text-xl font-bold">Human Evaluation</h1>
            <p className="text-sm text-gray-500">
              Rate each prediction on Relevance, Coherence, Naturalness (1-5) • Auto-saves to browser every 5 samples
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm font-medium">
                {completedCount} / {samples.length} ({progress}%)
              </p>
              {lastSaved && (
                <p className="text-xs text-gray-400">
                  Saved: {new Date(lastSaved).toLocaleTimeString()}
                </p>
              )}
            </div>
            <Button variant="outline" size="sm" onClick={saveSession}>
              <Save className="h-4 w-4 mr-1" />
              Save
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Progress bar */}
        <div className="bg-white rounded-lg p-2 shadow-sm">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Sample info */}
        <div className="flex items-center gap-2">
          <Badge variant="outline">Sample {currentIndex + 1} of {samples.length}</Badge>
          <Badge variant="secondary">{currentSample.dataset}</Badge>
          {existingRating && <Badge className="bg-green-100 text-green-800">Already rated</Badge>}
        </div>

        {/* Context Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Conversation Context (last 8 turns)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-[300px] overflow-y-auto space-y-2 pr-2">
                {currentSample.context.slice(-12).map((msg, i) => (
                  <div
                    key={i}
                    className={`p-3 rounded ${
                      msg.role === "user"
                        ? "bg-blue-50 border-l-3 border-blue-400"
                        : "bg-gray-100 border-l-3 border-gray-400"
                    }`}
                  >
                    <p className="text-xs font-bold text-gray-600 uppercase mb-1">{msg.role}</p>
                    <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-green-300 bg-green-50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-green-800">Ground Truth (Reference Only)</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-base whitespace-pre-wrap">{currentSample.groundTruth}</p>
              <p className="text-xs text-green-700 mt-3 italic">
                Note: Rate predictions on intrinsic quality, not similarity to ground truth. 
                A different but contextually appropriate response can score 5/5.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Predictions Row - 4 columns */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {currentSample.predictions.map((pred) => {
            const color = colors[pred.label] || colors.A;
            const predScores = scores[pred.label] || { relevance: 3, coherence: 3, naturalness: 3 };

            return (
              <Card key={pred.label} className={`border-2 ${color.border}`}>
                <CardHeader className={`pb-2 ${color.bg}`}>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <span className={`w-8 h-8 rounded-full ${color.circle} text-white flex items-center justify-center font-bold`}>
                      {pred.label}
                    </span>
                    Prediction {pred.label}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-3 space-y-4">
                  <div className="bg-white p-3 rounded border max-h-[150px] overflow-y-auto">
                    <p className="text-sm whitespace-pre-wrap">{pred.text}</p>
                  </div>

                  {/* Relevance */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-medium">Relevance</Label>
                      <span className={`text-xl font-bold ${color.text}`}>{predScores.relevance}</span>
                    </div>
                    <Slider
                      value={[predScores.relevance]}
                      onValueChange={(v) => updateScore(pred.label, "relevance", v[0])}
                      min={1}
                      max={5}
                      step={1}
                      className="h-3"
                    />
                  </div>

                  {/* Coherence */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-medium">Coherence</Label>
                      <span className={`text-xl font-bold ${color.text}`}>{predScores.coherence}</span>
                    </div>
                    <Slider
                      value={[predScores.coherence]}
                      onValueChange={(v) => updateScore(pred.label, "coherence", v[0])}
                      min={1}
                      max={5}
                      step={1}
                      className="h-3"
                    />
                  </div>

                  {/* Naturalness */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-medium">Naturalness</Label>
                      <span className={`text-xl font-bold ${color.text}`}>{predScores.naturalness}</span>
                    </div>
                    <Slider
                      value={[predScores.naturalness]}
                      onValueChange={(v) => updateScore(pred.label, "naturalness", v[0])}
                      min={1}
                      max={5}
                      step={1}
                      className="h-3"
                    />
                  </div>

                  <div className="text-center text-sm text-gray-500 pt-2 border-t">
                    Avg: <span className="font-bold">{((predScores.relevance + predScores.coherence + predScores.naturalness) / 3).toFixed(1)}</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Submit + Navigation */}
        <div className="flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
          <Button
            variant="ghost"
            onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
            disabled={currentIndex === 0}
          >
            ← Previous
          </Button>

          <Button onClick={handleSubmitRating} size="lg" className="px-8">
            {existingRating ? "Update & Next" : "Submit & Next"}
            <ChevronRight className="h-5 w-5 ml-2" />
          </Button>

          <Button
            variant="ghost"
            onClick={() => setCurrentIndex((i) => Math.min(samples.length - 1, i + 1))}
            disabled={currentIndex === samples.length - 1}
          >
            Skip →
          </Button>
        </div>

        {saving && (
          <div className="fixed bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg">
            Auto-saving...
          </div>
        )}
      </div>
    </div>
  );
}
