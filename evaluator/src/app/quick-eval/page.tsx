"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChevronRight, Save, Download, RotateCcw } from "lucide-react";
import { PredictionCard } from "@/components/prediction-card";

import {
  buildExportCsv,
  createDefaultScores,
  type PredictionRating,
  type Rating,
  type Sample,
} from "@/lib/quick-eval";


const STORAGE_KEY = "quick-eval-session-v4";
const AUTO_SAVE_INTERVAL = 5;
const COLOR_STYLES = [
  { border: "border-blue-200", bg: "bg-blue-50", text: "text-blue-600", circle: "bg-blue-600" },
  { border: "border-purple-200", bg: "bg-purple-50", text: "text-purple-600", circle: "bg-purple-600" },
  { border: "border-orange-200", bg: "bg-orange-50", text: "text-orange-600", circle: "bg-orange-600" },
  { border: "border-green-200", bg: "bg-green-50", text: "text-green-600", circle: "bg-green-600" },
  { border: "border-rose-200", bg: "bg-rose-50", text: "text-rose-600", circle: "bg-rose-600" },
  { border: "border-cyan-200", bg: "bg-cyan-50", text: "text-cyan-600", circle: "bg-cyan-600" },
];


export default function QuickEvalPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [scores, setScores] = useState<Record<string, PredictionRating>>({});
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
  const existingRating = currentSample ? ratings.find((rating) => rating.sampleId === currentSample.id) : null;

  useEffect(() => {
    if (!currentSample) {
      setScores({});
      return;
    }
    if (existingRating) {
      setScores(JSON.parse(JSON.stringify(existingRating.scores)));
    } else {
      setScores(createDefaultScores(currentSample.predictions));
    }
  }, [currentSample, existingRating]);

  const handleSubmitRating = () => {
    if (!currentSample) return;
    const newRating: Rating = {
      sampleId: currentSample.id,
      scores: JSON.parse(JSON.stringify(scores)),
      timestamp: new Date().toISOString(),
    };
    setRatings((prev) => [...prev.filter((rating) => rating.sampleId !== currentSample.id), newRating]);
    if (currentIndex < samples.length - 1) {
      setCurrentIndex((index) => index + 1);
    }
  };

  const handleExport = () => {
    const csv = buildExportCsv(samples, ratings);
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

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        <div className="flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
          <div>
            <h1 className="text-xl font-bold">Human Evaluation</h1>
            <p className="text-sm text-gray-500">
              Rate each prediction on Relevance, Coherence, Naturalness (1-5) • Auto-saves every 5 samples
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

        <div className="bg-white rounded-lg p-2 shadow-sm">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-blue-600 transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline">Sample {currentIndex + 1} of {samples.length}</Badge>
          <Badge variant="secondary">{currentSample.dataset}</Badge>
          {existingRating && <Badge className="bg-green-100 text-green-800">Already rated</Badge>}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Conversation Context</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-[300px] overflow-y-auto space-y-2 pr-2">
                {currentSample.context.slice(-12).map((msg, index) => (
                  <div
                    key={index}
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
                Rate intrinsic quality, not surface similarity. A different but contextually appropriate response can still score 5/5.
              </p>
            </CardContent>
          </Card>
        </div>

        <div
          className="grid gap-4"
          style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}
        >
          {currentSample.predictions.map((prediction, index) => {
            const color = COLOR_STYLES[index % COLOR_STYLES.length];
            const predictionScores = scores[prediction.label] || { relevance: 3, coherence: 3, naturalness: 3 };

            return (
              <PredictionCard
                key={prediction.label}
                color={color}
                prediction={prediction}
                scores={predictionScores}
                onUpdate={(category, value) => updateScore(prediction.label, category, value)}
              />
            );
          })}
        </div>

        <div className="flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
          <Button variant="ghost" onClick={() => setCurrentIndex((index) => Math.max(0, index - 1))} disabled={currentIndex === 0}>
            ← Previous
          </Button>

          <Button onClick={handleSubmitRating} size="lg" className="px-8">
            {existingRating ? "Update & Next" : "Submit & Next"}
            <ChevronRight className="h-5 w-5 ml-2" />
          </Button>

          <Button
            variant="ghost"
            onClick={() => setCurrentIndex((index) => Math.min(samples.length - 1, index + 1))}
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
