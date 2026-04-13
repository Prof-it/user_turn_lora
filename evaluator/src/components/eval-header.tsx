import { Download, RotateCcw, Save } from "lucide-react";

import { Button } from "@/components/ui/button";
import { KeyboardShortcuts } from "@/components/keyboard-shortcuts";
import type { RatingCategory } from "@/hooks/use-rating-keyboard";

interface EvalHeaderProps {
  completedCount: number;
  totalCount: number;
  progress: number;
  lastSaved: string | null;
  activeLabel: string;
  activeCategory: RatingCategory;
  onSave: () => void;
  onExport: () => void;
  onReset: () => void;
}

export function EvalHeader({
  completedCount,
  totalCount,
  progress,
  lastSaved,
  activeLabel,
  activeCategory,
  onSave,
  onExport,
  onReset,
}: EvalHeaderProps) {
  return (
    <div className="flex flex-col gap-3 bg-white rounded-lg p-4 shadow-sm xl:flex-row xl:items-center xl:justify-between">
      <div>
        <h1 className="text-xl font-bold">Human Evaluation</h1>
        <p className="text-sm text-gray-500">
          Rate each prediction on Relevance, Coherence, Naturalness (1-5) · Auto-saves every 5 samples
        </p>
      </div>
      <KeyboardShortcuts activeLabel={activeLabel} activeCategory={activeCategory} />
      <div className="flex items-center gap-4">
        <div className="text-right">
          <p className="text-sm font-medium">
            {completedCount} / {totalCount} ({progress}%)
          </p>
          {lastSaved && (
            <p className="text-xs text-gray-400">
              Saved: {new Date(lastSaved).toLocaleTimeString()}
            </p>
          )}
        </div>
        <Button variant="outline" size="sm" onClick={onSave}>
          <Save className="h-4 w-4 mr-1" />
          Save
        </Button>
        <Button variant="outline" size="sm" onClick={onExport}>
          <Download className="h-4 w-4 mr-1" />
          Export
        </Button>
        <Button variant="ghost" size="sm" onClick={onReset} aria-label="Reset ratings">
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
