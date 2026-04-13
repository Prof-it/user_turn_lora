import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface EvalCompleteCardProps {
  totalCount: number;
  onExport: () => void;
}

export function EvalCompleteCard({ totalCount, onExport }: EvalCompleteCardProps) {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <Card className="max-w-md">
        <CardContent className="p-8 text-center">
          <h2 className="text-xl font-bold mb-4">Evaluation Complete!</h2>
          <p className="text-gray-600 mb-4">You have rated all {totalCount} samples.</p>
          <Button onClick={onExport}>
            <Download className="h-4 w-4 mr-2" />
            Export Results
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
