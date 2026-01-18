"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle } from "lucide-react";

type ErrorType = "wrong_extraction" | "truncated" | "noise" | "other";

interface LabelValidationFormProps {
  onSubmit: (validation: {
    isValid: boolean;
    errorType?: ErrorType;
    notes?: string;
  }) => void;
  initialValues?: {
    isValid: boolean;
    errorType?: ErrorType;
    notes?: string;
  };
}

export function LabelValidationForm({ onSubmit, initialValues }: LabelValidationFormProps) {
  const [isValid, setIsValid] = useState<boolean | null>(initialValues?.isValid ?? null);
  const [errorType, setErrorType] = useState<ErrorType | undefined>(initialValues?.errorType);
  const [notes, setNotes] = useState(initialValues?.notes || "");
  const [submitted, setSubmitted] = useState(!!initialValues);

  const handleSubmit = () => {
    if (isValid === null) return;
    onSubmit({
      isValid,
      errorType: isValid ? undefined : errorType,
      notes: notes || undefined,
    });
    setSubmitted(true);
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Label Validation</CardTitle>
        <p className="text-sm text-muted-foreground">
          Is the extracted (context → next user turn) label correct?
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-3">
          <Button
            variant={isValid === true ? "default" : "outline"}
            className={`flex-1 ${isValid === true ? "bg-green-600 hover:bg-green-700" : ""}`}
            onClick={() => setIsValid(true)}
          >
            <CheckCircle className="mr-2 h-4 w-4" />
            Valid
          </Button>
          <Button
            variant={isValid === false ? "default" : "outline"}
            className={`flex-1 ${isValid === false ? "bg-red-600 hover:bg-red-700" : ""}`}
            onClick={() => setIsValid(false)}
          >
            <XCircle className="mr-2 h-4 w-4" />
            Invalid
          </Button>
        </div>

        {isValid === false && (
          <div className="space-y-2">
            <Label className="text-sm font-medium">Error Type</Label>
            <Select value={errorType || ""} onValueChange={(v) => setErrorType(v as ErrorType)}>
              <SelectTrigger>
                <SelectValue placeholder="Select error type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="wrong_extraction">Wrong Extraction</SelectItem>
                <SelectItem value="truncated">Truncated</SelectItem>
                <SelectItem value="noise">Noise/Garbage</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="space-y-2">
          <Label className="text-sm font-medium">Notes (optional)</Label>
          <Textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Describe the issue if invalid..."
            className="h-16"
          />
        </div>

        <Button
          onClick={handleSubmit}
          disabled={isValid === null}
          className="w-full"
        >
          {submitted ? "✓ Saved" : "Submit Validation"}
        </Button>

        {submitted && (
          <div className="flex justify-center">
            <Badge variant="default" className="bg-green-600">
              ✓ Validation saved
            </Badge>
          </div>
        )}

        {!submitted && isValid !== null && (
          <div className="flex justify-center">
            <Badge variant={isValid ? "default" : "destructive"}>
              {isValid ? "Marked as Valid" : "Marked as Invalid"}
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
