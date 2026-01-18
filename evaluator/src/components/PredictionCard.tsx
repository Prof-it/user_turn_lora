"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface PredictionCardProps {
  label: string;
  prediction: string;
  isSelected?: boolean;
  onClick?: () => void;
}

export function PredictionCard({
  label,
  prediction,
  isSelected,
  onClick,
}: PredictionCardProps) {
  return (
    <Card
      className={`cursor-pointer transition-all ${
        isSelected
          ? "ring-2 ring-primary border-primary"
          : "hover:border-primary/50"
      }`}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <Badge variant="outline" className="text-lg px-3 py-1">
            {label}
          </Badge>
          {isSelected && (
            <Badge className="bg-primary">Selected</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px]">
          <p className="text-sm whitespace-pre-wrap">
            {prediction || <span className="text-muted-foreground italic">No prediction available</span>}
          </p>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
