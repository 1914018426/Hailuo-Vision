import { useState, useCallback } from 'react';
import { GripVertical } from 'lucide-react';
import { cn } from '@/lib/utils';
import { VideoPanel } from './VideoPanel';
import type { DetectionResult, DisplayConfig } from '@/types';

interface VideoGridProps {
  frames: Record<string, string>;
  detections: Record<string, DetectionResult>;
  displayConfig: DisplayConfig;
  onReorder: (newOrder: string[]) => void;
}

function getGridCols(count: number): string {
  if (count <= 1) return 'grid-cols-1';
  if (count === 2) return 'grid-cols-1 md:grid-cols-2';
  if (count <= 4) return 'grid-cols-1 md:grid-cols-2';
  if (count <= 6) return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
  return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4';
}

export function VideoGrid({
  frames,
  detections,
  displayConfig,
  onReorder,
}: VideoGridProps) {
  const { order, labels } = displayConfig;
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const handleDragStart = useCallback(
    (index: number) => (e: React.DragEvent) => {
      setDraggedIndex(index);
      e.dataTransfer.effectAllowed = 'move';
      // 必须设置 dataTransfer，否则 Firefox 不触发 drag
      e.dataTransfer.setData('text/plain', String(index));
      // 延迟添加 dragged 样式，避免拖拽 ghost 也受影响
      const el = e.currentTarget as HTMLElement;
      requestAnimationFrame(() => {
        el.classList.add('opacity-40');
      });
    },
    []
  );

  const handleDragEnd = useCallback((e: React.DragEvent) => {
    (e.currentTarget as HTMLElement).classList.remove('opacity-40');
    setDraggedIndex(null);
    setDragOverIndex(null);
  }, []);

  const handleDragOver = useCallback(
    (index: number) => (e: React.DragEvent) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      if (draggedIndex !== null && draggedIndex !== index) {
        setDragOverIndex(index);
      }
    },
    [draggedIndex]
  );

  const handleDragLeave = useCallback(() => {
    setDragOverIndex(null);
  }, []);

  const handleDrop = useCallback(
    (targetIndex: number) => (e: React.DragEvent) => {
      e.preventDefault();
      const sourceIndex = Number(e.dataTransfer.getData('text/plain'));
      if (
        !Number.isNaN(sourceIndex) &&
        sourceIndex !== targetIndex &&
        sourceIndex >= 0 &&
        sourceIndex < order.length
      ) {
        const newOrder = [...order];
        const [moved] = newOrder.splice(sourceIndex, 1);
        newOrder.splice(targetIndex, 0, moved);
        onReorder(newOrder);
      }
      setDraggedIndex(null);
      setDragOverIndex(null);
    },
    [order, onReorder]
  );

  return (
    <div
      className={cn(
        'grid gap-3 p-3 md:gap-4 md:p-4',
        getGridCols(order.length)
      )}
    >
      {order.map((cameraId, index) => (
        <div
          key={cameraId}
          draggable
          onDragStart={handleDragStart(index)}
          onDragEnd={handleDragEnd}
          onDragOver={handleDragOver(index)}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop(index)}
          className={cn(
            'relative group cursor-move transition-all duration-300',
            dragOverIndex === index && draggedIndex !== null &&
              draggedIndex !== index &&
              'scale-[1.02] ring-2 ring-teal-500/50 rounded-xl z-10',
            draggedIndex === index && 'opacity-40'
          )}
        >
          {/* Drag handle */}
          <div className="absolute top-2 right-2 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
            <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-black/50 backdrop-blur-sm border border-white/10">
              <GripVertical className="w-3.5 h-3.5 text-slate-300" />
              <span className="text-[10px] text-slate-300">拖动排序</span>
            </div>
          </div>

          <VideoPanel
            cameraId={cameraId}
            label={labels[cameraId] || cameraId}
            frameImage={frames[cameraId]}
            detection={detections[cameraId] || {
              camera_id: cameraId,
              person_count: 0,
              detections: [],
              best_gesture: 'none',
              best_gesture_confidence: 0,
              inference_ms: 0,
              timestamp: 0,
            }}
          />
        </div>
      ))}
    </div>
  );
}
