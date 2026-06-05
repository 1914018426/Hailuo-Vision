import { cn } from '@/lib/utils';
import { GESTURE_LABELS } from '@/types';
import type { DetectionResult, Gesture } from '@/types';

interface GestureOverlayProps {
  detection: DetectionResult;
}

function getGestureBadgeColor(gesture: Gesture): string {
  switch (gesture) {
    case 'hand_up':
      return 'bg-slate-500/80 text-white';
    case 'waving':
      return 'bg-red-500/80 text-white';
    case 'checking':
      return 'bg-orange-500/80 text-white';
    default:
      return 'bg-slate-600/60 text-slate-300';
  }
}

export function GestureOverlay({ detection }: GestureOverlayProps) {
  // 始终渲染占位，避免 none/checking 切换时高度跳变导致画面抖动
  const isNone = detection.best_gesture === 'none';
  return (
    <div
      className={cn(
        'absolute top-2 right-2 z-10 pointer-events-none',
        'px-2 py-0.5 rounded-md text-[11px] font-semibold',
        isNone ? 'opacity-0' : 'opacity-100 animate-fade-in',
        getGestureBadgeColor(detection.best_gesture)
      )}
    >
      {GESTURE_LABELS[detection.best_gesture] ?? ''}
    </div>
  );
}
