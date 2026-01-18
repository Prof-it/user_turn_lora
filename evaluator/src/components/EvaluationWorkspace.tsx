"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConversationDisplay } from "./ConversationDisplay";
import { PredictionCard } from "./PredictionCard";
import { RatingForm } from "./RatingForm";
import { LabelValidationForm } from "./LabelValidationForm";
import { ProgressBar } from "./ProgressBar";
import {
  EvaluationSample,
  EvaluationSession,
  ModelData,
  IssueTag,
} from "@/types/evaluation";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Save,
  BarChart3,
} from "lucide-react";

interface EvaluationWorkspaceProps {
  session: EvaluationSession;
  samples: EvaluationSample[];
  models: ModelData[];
  onSessionUpdate: (session: EvaluationSession) => void;
}

export function EvaluationWorkspace({
  session,
  samples,
  models,
  onSessionUpdate,
}: EvaluationWorkspaceProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedPrediction, setSelectedPrediction] = useState<string | null>(null);
  const [ratedPredictions, setRatedPredictions] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);

  const currentSample = samples[currentIndex];

  // Build blinded predictions list
  const blindedPredictions = currentSample
    ? Object.entries(currentSample.blindedMapping).map(([label, mapping]) => {
        const [modelId, type] = mapping.split(":");
        const prediction = currentSample.predictions.find((p) => p.modelId === modelId);
        return {
          label,
          modelId,
          type: type as "base" | "fineTuned",
          text: type === "base" ? prediction?.base : prediction?.fineTuned,
        };
      })
    : [];

  // Check which predictions have been rated for current sample
  useEffect(() => {
    if (!currentSample) return;
    const rated = new Set<string>();
    for (const rating of session.ratings) {
      if (rating.sampleId === currentSample.id) {
        rated.add(rating.blindedLabel);
      }
    }
    setRatedPredictions(rated);
  }, [currentSample, session.ratings]);

  const handleRatingSubmit = useCallback(
    async (
      blindedLabel: string,
      ratings: {
        relevance: number;
        coherence: number;
        naturalness: number;
        specificity?: number;
        issueTag?: IssueTag;
        notes?: string;
      }
    ) => {
      if (!currentSample) return;

      const mapping = currentSample.blindedMapping[blindedLabel];
      const [modelId, predictionType] = mapping.split(":");

      setSaving(true);
      try {
        const response = await fetch(`/api/sessions/${session.id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "addRating",
            rating: {
              sampleId: currentSample.id,
              blindedLabel,
              modelId,
              predictionType,
              ...ratings,
              raterId: session.raterId,
            },
          }),
        });

        if (response.ok) {
          const data = await response.json();
          onSessionUpdate(data.session);
          setRatedPredictions((prev) => new Set([...prev, blindedLabel]));
        }
      } catch (error) {
        console.error("Failed to save rating:", error);
      } finally {
        setSaving(false);
      }
    },
    [currentSample, session.id, session.raterId, onSessionUpdate]
  );

  const handleValidationSubmit = useCallback(
    async (validation: {
      isValid: boolean;
      errorType?: string;
      notes?: string;
    }) => {
      if (!currentSample) return;

      setSaving(true);
      try {
        const response = await fetch(`/api/sessions/${session.id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "addValidation",
            validation: {
              sampleId: currentSample.id,
              ...validation,
              raterId: session.raterId,
            },
          }),
        });

        if (response.ok) {
          const data = await response.json();
          onSessionUpdate(data.session);
        }
      } catch (error) {
        console.error("Failed to save validation:", error);
      } finally {
        setSaving(false);
      }
    },
    [currentSample, session.id, session.raterId, onSessionUpdate]
  );

  const handleExport = async () => {
    try {
      const response = await fetch(`/api/sessions/${session.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "export" }),
      });

      if (response.ok) {
        const data = await response.json();

        // Download ratings CSV
        const ratingsBlob = new Blob([data.ratingsCSV], { type: "text/csv" });
        const ratingsUrl = URL.createObjectURL(ratingsBlob);
        const ratingsLink = document.createElement("a");
        ratingsLink.href = ratingsUrl;
        ratingsLink.download = `human_eval_ratings_${session.id}.csv`;
        ratingsLink.click();

        // Download validations CSV
        if (data.validationsCSV) {
          const validationsBlob = new Blob([data.validationsCSV], { type: "text/csv" });
          const validationsUrl = URL.createObjectURL(validationsBlob);
          const validationsLink = document.createElement("a");
          validationsLink.href = validationsUrl;
          validationsLink.download = `label_validations_${session.id}.csv`;
          validationsLink.click();
        }
      }
    } catch (error) {
      console.error("Failed to export:", error);
    }
  };

  const handleExportToModels = async () => {
    try {
      await fetch(`/api/sessions/${session.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "exportToModels" }),
      });
      alert("Ratings exported to model folders!");
    } catch (error) {
      console.error("Failed to export to models:", error);
    }
  };

  const currentValidation = session.labelValidations.find(
    (v) => v.sampleId === currentSample?.id
  );

  const existingRating = selectedPrediction
    ? session.ratings.find(
        (r) => r.sampleId === currentSample?.id && r.blindedLabel === selectedPrediction
      )
    : undefined;

  if (!currentSample) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <p className="text-muted-foreground">No samples available for evaluation.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with progress */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="text-sm">
                Sample {currentIndex + 1} of {samples.length}
              </Badge>
              <Badge variant="secondary">{currentSample.dataset}</Badge>
              <Badge variant="outline">ID: {currentSample.id}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleExport}>
                <Download className="h-4 w-4 mr-1" />
                Export CSV
              </Button>
              <Button variant="outline" size="sm" onClick={handleExportToModels}>
                <Save className="h-4 w-4 mr-1" />
                Save to Models
              </Button>
            </div>
          </div>
          <ProgressBar
            completed={session.progress.completed}
            total={session.progress.total}
            label="Overall Progress"
          />
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
          disabled={currentIndex === 0}
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Previous
        </Button>
        <Select
          value={String(currentIndex)}
          onValueChange={(v) => setCurrentIndex(parseInt(v))}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {samples.map((sample, i) => (
              <SelectItem key={sample.id} value={String(i)}>
                {i + 1}. {sample.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          onClick={() => setCurrentIndex((i) => Math.min(samples.length - 1, i + 1))}
          disabled={currentIndex === samples.length - 1}
        >
          Next
          <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Context and Ground Truth */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Conversation Context</CardTitle>
            </CardHeader>
            <CardContent>
              <ConversationDisplay
                messages={currentSample.context}
                groundTruth={currentSample.groundTruth}
                maxHeight="500px"
              />
            </CardContent>
          </Card>

          <LabelValidationForm
            key={currentSample.id}
            onSubmit={handleValidationSubmit}
            initialValues={
              currentValidation
                ? {
                    isValid: currentValidation.isValid,
                    errorType: currentValidation.errorType as "wrong_extraction" | "truncated" | "noise" | "other" | undefined,
                    notes: currentValidation.notes,
                  }
                : undefined
            }
          />
        </div>

        {/* Right: Predictions and Rating */}
        <div className="space-y-4">
          <Tabs defaultValue="predictions">
            <TabsList className="w-full">
              <TabsTrigger value="predictions" className="flex-1">
                Predictions ({blindedPredictions.length})
              </TabsTrigger>
              <TabsTrigger value="stats" className="flex-1">
                <BarChart3 className="h-4 w-4 mr-1" />
                Stats
              </TabsTrigger>
            </TabsList>

            <TabsContent value="predictions" className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {blindedPredictions.map((pred) => (
                  <PredictionCard
                    key={pred.label}
                    label={pred.label}
                    prediction={pred.text || ""}
                    isSelected={selectedPrediction === pred.label}
                    onClick={() => setSelectedPrediction(pred.label)}
                  />
                ))}
              </div>

              <div className="flex flex-wrap gap-2">
                {blindedPredictions.map((pred) => (
                  <Badge
                    key={pred.label}
                    variant={ratedPredictions.has(pred.label) ? "default" : "outline"}
                  >
                    {pred.label}: {ratedPredictions.has(pred.label) ? "✓ Rated" : "Pending"}
                  </Badge>
                ))}
              </div>

              <Separator />

              {selectedPrediction ? (
                <RatingForm
                  blindedLabel={selectedPrediction}
                  onSubmit={(ratings) => handleRatingSubmit(selectedPrediction, ratings)}
                  initialValues={
                    existingRating
                      ? {
                          relevance: existingRating.relevance,
                          coherence: existingRating.coherence,
                          naturalness: existingRating.naturalness,
                          specificity: existingRating.specificity,
                          issueTag: existingRating.issueTag as IssueTag,
                          notes: existingRating.notes,
                        }
                      : undefined
                  }
                />
              ) : (
                <Card>
                  <CardContent className="p-8 text-center text-muted-foreground">
                    Select a prediction (A, B, C...) above to rate it
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="stats">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Session Statistics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Total Ratings</p>
                      <p className="text-2xl font-bold">{session.ratings.length}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Samples Completed</p>
                      <p className="text-2xl font-bold">{session.progress.completed}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Label Validations</p>
                      <p className="text-2xl font-bold">{session.labelValidations.length}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Invalid Labels</p>
                      <p className="text-2xl font-bold text-red-500">
                        {session.labelValidations.filter((v) => !v.isValid).length}
                      </p>
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <p className="text-sm font-medium mb-2">Models in Session</p>
                    <div className="flex flex-wrap gap-2">
                      {models.map((model) => (
                        <Badge key={model.id} variant="outline">
                          {model.name}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {saving && (
        <div className="fixed bottom-4 right-4 bg-primary text-primary-foreground px-4 py-2 rounded-md shadow-lg">
          Saving...
        </div>
      )}
    </div>
  );
}
