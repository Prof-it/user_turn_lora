"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { RATING_CATEGORIES, ISSUE_TAGS, IssueTag } from "@/types/evaluation";
import { Info } from "lucide-react";

interface RatingFormProps {
  blindedLabel: string;
  onSubmit: (ratings: {
    relevance: number;
    coherence: number;
    naturalness: number;
    specificity?: number;
    issueTag?: IssueTag;
    notes?: string;
  }) => void;
  initialValues?: {
    relevance: number;
    coherence: number;
    naturalness: number;
    specificity?: number;
    issueTag?: IssueTag;
    notes?: string;
  };
}

export function RatingForm({ blindedLabel, onSubmit, initialValues }: RatingFormProps) {
  const [relevance, setRelevance] = useState(initialValues?.relevance || 3);
  const [coherence, setCoherence] = useState(initialValues?.coherence || 3);
  const [naturalness, setNaturalness] = useState(initialValues?.naturalness || 3);
  const [specificity, setSpecificity] = useState(initialValues?.specificity || 3);
  const [includeSpecificity, setIncludeSpecificity] = useState(!!initialValues?.specificity);
  const [issueTag, setIssueTag] = useState<IssueTag | undefined>(initialValues?.issueTag);
  const [notes, setNotes] = useState(initialValues?.notes || "");
  const [showExamples, setShowExamples] = useState<string | null>(null);

  const handleSubmit = () => {
    onSubmit({
      relevance,
      coherence,
      naturalness,
      specificity: includeSpecificity ? specificity : undefined,
      issueTag,
      notes: notes || undefined,
    });
  };

  const renderSlider = (
    category: (typeof RATING_CATEGORIES)[number],
    value: number,
    setValue: (v: number) => void
  ) => (
    <div key={category.id} className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium flex items-center gap-2">
          {category.name}
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            onClick={() => setShowExamples(showExamples === category.id ? null : category.id)}
          >
            <Info className="h-3 w-3" />
          </Button>
        </Label>
        <Badge variant="outline" className="text-lg px-3">
          {value}
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground">{category.description}</p>
      {showExamples === category.id && (
        <div className="text-xs bg-muted p-2 rounded space-y-1">
          <p><strong>1:</strong> {category.examples.score1}</p>
          <p><strong>3:</strong> {category.examples.score3}</p>
          <p><strong>5:</strong> {category.examples.score5}</p>
        </div>
      )}
      <Slider
        value={[value]}
        onValueChange={(v) => setValue(v[0])}
        min={1}
        max={5}
        step={1}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Poor (1)</span>
        <span>Average (3)</span>
        <span>Excellent (5)</span>
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          Rate Prediction
          <Badge variant="outline" className="text-lg px-3">
            {blindedLabel}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {renderSlider(RATING_CATEGORIES[0], relevance, setRelevance)}
        {renderSlider(RATING_CATEGORIES[1], coherence, setCoherence)}
        {renderSlider(RATING_CATEGORIES[2], naturalness, setNaturalness)}

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="includeSpecificity"
            checked={includeSpecificity}
            onChange={(e) => setIncludeSpecificity(e.target.checked)}
            className="rounded"
          />
          <Label htmlFor="includeSpecificity" className="text-sm">
            Include Specificity/Usefulness rating (optional)
          </Label>
        </div>

        {includeSpecificity && renderSlider(RATING_CATEGORIES[3], specificity, setSpecificity)}

        <div className="space-y-2">
          <Label className="text-sm font-medium">Issue Tag (optional)</Label>
          <Select value={issueTag || "none"} onValueChange={(v) => setIssueTag(v === "none" ? undefined : v as IssueTag)}>
            <SelectTrigger>
              <SelectValue placeholder="Select an issue tag if applicable" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              {ISSUE_TAGS.map((tag) => (
                <SelectItem key={tag} value={tag}>
                  {tag}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label className="text-sm font-medium">Notes (optional)</Label>
          <Textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Any additional observations..."
            className="h-20"
          />
        </div>

        <Button onClick={handleSubmit} className="w-full">
          Submit Rating for {blindedLabel}
        </Button>
      </CardContent>
    </Card>
  );
}
