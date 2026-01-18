"use client";

import { useEffect, useRef } from "react";
import { Message } from "@/types/evaluation";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ConversationDisplayProps {
  messages: Message[];
  groundTruth?: string;
  maxHeight?: string;
}

export function ConversationDisplay({
  messages,
  groundTruth,
  maxHeight = "500px",
}: ConversationDisplayProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to bottom when messages change
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div 
      ref={scrollRef}
      className="w-full rounded-md border overflow-y-auto"
      style={{ maxHeight }}
    >
      <div className="p-4 space-y-3">
        {messages.map((message, index) => (
          <div
            key={index}
            className={cn(
              "p-3 rounded-lg",
              message.role === "user"
                ? "bg-blue-50 dark:bg-blue-950 ml-4"
                : message.role === "assistant"
                ? "bg-gray-50 dark:bg-gray-900 mr-4"
                : "bg-yellow-50 dark:bg-yellow-950"
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <Badge
                variant={
                  message.role === "user"
                    ? "default"
                    : message.role === "assistant"
                    ? "secondary"
                    : "outline"
                }
                className="text-xs"
              >
                {message.role}
              </Badge>
              <span className="text-xs text-muted-foreground">
                Turn {index + 1}
              </span>
            </div>
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          </div>
        ))}

        {groundTruth && (
          <div className="p-3 rounded-lg bg-green-50 dark:bg-green-950 ml-4 border-2 border-green-200 dark:border-green-800">
            <div className="flex items-center gap-2 mb-1">
              <Badge className="bg-green-600 text-xs">Ground Truth (Next User Turn)</Badge>
            </div>
            <p className="text-sm whitespace-pre-wrap">{groundTruth}</p>
          </div>
        )}
      </div>
    </div>
  );
}
