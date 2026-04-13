import type { RatingCategory } from "@/hooks/use-rating-keyboard";

interface KeyboardShortcutsProps {
  activeLabel: string;
  activeCategory: RatingCategory;
}

function Key({ children }: { children: string }) {
  return <kbd className="rounded border bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-700">{children}</kbd>;
}

export function KeyboardShortcuts({ activeLabel, activeCategory }: KeyboardShortcutsProps) {
  return (
    <div className="rounded-lg border bg-slate-50 px-3 py-2 text-xs text-gray-600">
      <div className="font-medium text-gray-800">
        Active: {activeLabel} · {activeCategory}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
        <span><Key>↑</Key>/<Key>↓</Key> field</span>
        <span><Key>←</Key>/<Key>→</Key> score</span>
        <span><Key>1</Key>-<Key>5</Key> set</span>
        <span><Key>Enter</Key> next</span>
        <span><Key>P</Key> prev</span>
        <span><Key>S</Key> skip</span>
      </div>
    </div>
  );
}
