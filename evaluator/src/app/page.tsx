"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { EvaluationWorkspace } from "@/components/EvaluationWorkspace";
import {
  ModelData,
  EvaluationSample,
  EvaluationSession,
  DEFAULT_SAMPLING_CONFIG,
} from "@/types/evaluation";
import { Play, Settings, Users, Database, FileText } from "lucide-react";

type AppState = "setup" | "evaluating";

interface CorpusStats {
  totalSamples: number;
  evalSamples: number;
  trainingSamples: number;
  datasetBreakdown: Record<string, number>;
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>("setup");
  const [models, setModels] = useState<ModelData[]>([]);
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [raterId, setRaterId] = useState("");
  const [seed, setSeed] = useState(DEFAULT_SAMPLING_CONFIG.seed);
  const [samplePercentage, setSamplePercentage] = useState(DEFAULT_SAMPLING_CONFIG.samplePercentage);
  const [includeTraining, setIncludeTraining] = useState(true);
  const [loading, setLoading] = useState(true);
  const [samples, setSamples] = useState<EvaluationSample[]>([]);
  const [session, setSession] = useState<EvaluationSession | null>(null);
  const [existingSessions, setExistingSessions] = useState<
    { id: string; raterId: string; startedAt: string; progress: { completed: number; total: number } }[]
  >([]);

  // Load models and existing sessions on mount
  useEffect(() => {
    async function loadData() {
      try {
        const [modelsRes, sessionsRes] = await Promise.all([
          fetch("/api/models"),
          fetch("/api/sessions"),
        ]);

        if (modelsRes.ok) {
          const data = await modelsRes.json();
          setModels(data.models);
          setStats(data.stats);
          setSelectedModels(data.models.map((m: ModelData) => m.id));
        }

        if (sessionsRes.ok) {
          const data = await sessionsRes.json();
          setExistingSessions(data.sessions);
        }
      } catch (error) {
        console.error("Failed to load data:", error);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const handleStartEvaluation = async () => {
    if (!raterId.trim()) {
      alert("Please enter a Rater ID");
      return;
    }

    if (selectedModels.length === 0) {
      alert("Please select at least one model");
      return;
    }

    setLoading(true);

    try {
      // Load samples
      const samplesRes = await fetch(
        `/api/samples?models=${selectedModels.join(",")}&seed=${seed}&samplePercentage=${samplePercentage}&includeTraining=${includeTraining}`
      );

      if (!samplesRes.ok) {
        throw new Error("Failed to load samples");
      }

      const samplesData = await samplesRes.json();
      setSamples(samplesData.samples);

      // Create session
      const sessionRes = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raterId,
          selectedModels,
          sampleIds: samplesData.samples.map((s: EvaluationSample) => s.id),
          seed,
        }),
      });

      if (!sessionRes.ok) {
        throw new Error("Failed to create session");
      }

      const sessionData = await sessionRes.json();
      setSession(sessionData.session);
      setAppState("evaluating");
    } catch (error) {
      console.error("Failed to start evaluation:", error);
      alert("Failed to start evaluation. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const handleResumeSession = async (sessionId: string) => {
    setLoading(true);

    try {
      const sessionRes = await fetch(`/api/sessions/${sessionId}`);
      if (!sessionRes.ok) {
        throw new Error("Failed to load session");
      }

      const sessionData = await sessionRes.json();
      setSession(sessionData.session);

      // Load samples with the same config
      const samplesRes = await fetch(
        `/api/samples?models=${sessionData.session.selectedModels.join(",")}&seed=${sessionData.session.seed}&samplePercentage=${samplePercentage}&includeTraining=${includeTraining}`
      );

      if (!samplesRes.ok) {
        throw new Error("Failed to load samples");
      }

      const samplesData = await samplesRes.json();
      setSamples(samplesData.samples);
      setAppState("evaluating");
    } catch (error) {
      console.error("Failed to resume session:", error);
      alert("Failed to resume session. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const estimatedSamples = stats
    ? Math.ceil(
        (includeTraining ? stats.totalSamples : stats.evalSamples) *
          (samplePercentage / 100)
      )
    : 0;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (appState === "evaluating" && session && samples.length > 0) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold">Human Evaluation</h1>
              <p className="text-muted-foreground">
                Session: {session.id} | Rater: {session.raterId}
              </p>
            </div>
            <Button variant="outline" onClick={() => setAppState("setup")}>
              Back to Setup
            </Button>
          </div>
          <EvaluationWorkspace
            session={session}
            samples={samples}
            models={models.filter((m) => selectedModels.includes(m.id))}
            onSessionUpdate={setSession}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Human Evaluation Framework</h1>
          <p className="text-muted-foreground">
            Evaluate model predictions with blinded A/B comparison and structured rubrics
          </p>
        </div>

        {/* Corpus Stats */}
        {stats && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Corpus Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Total Samples</p>
                  <p className="text-2xl font-bold">{stats.totalSamples.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Eval Samples</p>
                  <p className="text-2xl font-bold">{stats.evalSamples.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Training Samples</p>
                  <p className="text-2xl font-bold">{stats.trainingSamples.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Datasets</p>
                  <div className="flex gap-2 mt-1">
                    {Object.entries(stats.datasetBreakdown).map(([name, count]) => (
                      <Badge key={name} variant="outline">
                        {name}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Model Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Model Selection
            </CardTitle>
            <CardDescription>
              Select which models to include in the evaluation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {models.map((model) => (
                <div
                  key={model.id}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedModels.includes(model.id)
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  }`}
                  onClick={() => toggleModelSelection(model.id)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{model.name}</p>
                      <p className="text-xs text-muted-foreground">{model.id}</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model.id)}
                      onChange={() => toggleModelSelection(model.id)}
                      className="rounded"
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Sampling Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Sampling Configuration
            </CardTitle>
            <CardDescription>
              Configure reproducible stratified sampling (≥5% recommended)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label>Sample Percentage: {samplePercentage}%</Label>
                <Slider
                  value={[samplePercentage]}
                  onValueChange={(v) => setSamplePercentage(v[0])}
                  min={1}
                  max={100}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  ≈ {estimatedSamples} samples to evaluate
                </p>
              </div>

              <div className="space-y-2">
                <Label>Random Seed</Label>
                <Select value={String(seed)} onValueChange={(v) => setSeed(parseInt(v))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="42">42 (default)</SelectItem>
                    <SelectItem value="123">123</SelectItem>
                    <SelectItem value="456">456</SelectItem>
                    <SelectItem value="789">789</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="includeTraining"
                checked={includeTraining}
                onChange={(e) => setIncludeTraining(e.target.checked)}
                className="rounded"
              />
              <Label htmlFor="includeTraining">
                Include training samples (recommended for full corpus coverage)
              </Label>
            </div>
          </CardContent>
        </Card>

        {/* Rater Setup */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Rater Setup
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Rater ID</Label>
              <input
                type="text"
                value={raterId}
                onChange={(e) => setRaterId(e.target.value)}
                placeholder="Enter your name or ID"
                className="w-full px-3 py-2 border rounded-md"
              />
            </div>

            <Button
              onClick={handleStartEvaluation}
              disabled={!raterId.trim() || selectedModels.length === 0}
              className="w-full"
              size="lg"
            >
              <Play className="h-4 w-4 mr-2" />
              Start New Evaluation Session
            </Button>
          </CardContent>
        </Card>

        {/* Existing Sessions */}
        {existingSessions.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Resume Existing Session</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {existingSessions.map((s) => (
                  <div
                    key={s.id}
                    className="flex items-center justify-between p-3 border rounded-lg"
                  >
                    <div>
                      <p className="font-medium">{s.raterId}</p>
                      <p className="text-xs text-muted-foreground">
                        Started: {new Date(s.startedAt).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant="outline">
                        {s.progress.completed}/{s.progress.total} (
                        {Math.round((s.progress.completed / s.progress.total) * 100) || 0}%)
                      </Badge>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleResumeSession(s.id)}
                      >
                        Resume
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Separator />

        {/* Calibration Info */}
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Guidelines</CardTitle>
          </CardHeader>
          <CardContent className="prose prose-sm dark:prose-invert max-w-none">
            <h4>Rating Categories (1-5 scale)</h4>
            <ul>
              <li>
                <strong>Relevance:</strong> Does the user turn directly respond to the
                assistant&apos;s last message and remain on-topic?
              </li>
              <li>
                <strong>Coherence:</strong> Is it logically consistent with the prior turns
                and doesn&apos;t contradict the conversation?
              </li>
              <li>
                <strong>Naturalness:</strong> Does it sound like a plausible human user
                message (not assistant-y, not overly verbose)?
              </li>
              <li>
                <strong>Specificity (optional):</strong> Does it move the conversation
                forward with meaningful info/question?
              </li>
            </ul>

            <h4>Label Validation</h4>
            <p>
              For each sample, also validate whether the extracted (context → next user
              turn) label is correct. Mark as invalid if the extraction is wrong,
              truncated, or contains noise.
            </p>

            <h4>Blinded Evaluation</h4>
            <p>
              Predictions are labeled A, B, C, etc. with randomized order. The actual
              model identity is hidden during rating and revealed in the export.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
