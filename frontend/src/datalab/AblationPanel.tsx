import { useState } from 'react';
import {
  Play,
  Square,
  Trash2,
  RefreshCw,
  Beaker,
  CheckCircle2,
  XCircle,
  Loader2,
  FlaskConical,
} from 'lucide-react';
import { useRecordings, useExperiments, useExperimentControl, useDataLabWebSocket } from './useDataLab';
import type { AblationExperiment, ExperimentType } from './types';

const ENGINE_OPTIONS = [
  { key: 'simple', label: 'Simple' },
  { key: 'simple_minicpm', label: 'Simple+MiniCPM ★' },
  { key: 'minicpm', label: 'MiniCPM' },
];

const EXPERIMENT_TYPE_OPTIONS: { key: ExperimentType; label: string; desc: string }[] = [
  {
    key: 'engine_comparison',
    label: '引擎横向对比',
    desc: '3 个主引擎在相同输入下的表现对比',
  },
  {
    key: 'component_ablation',
    label: '组件消融',
    desc: '逐一移除 Simple+MiniCPM 的组件，量化各组件贡献',
  },
  {
    key: 'threshold_sweep',
    label: '阈值扫描',
    desc: '扫描 0.3~0.9 置信度阈值，生成 PR/ROC 曲线',
  },
  {
    key: 'scenario_analysis',
    label: '场景分析',
    desc: '按速度/距离/左右手分场景统计各引擎表现',
  },
];

const STATUS_LABELS: Record<string, { text: string; className: string }> = {
  pending: { text: '等待中', className: 'bg-slate-500/20 text-slate-400' },
  running: { text: '运行中', className: 'bg-sky-500/20 text-sky-400' },
  completed: { text: '已完成', className: 'bg-emerald-500/20 text-emerald-400' },
  failed: { text: '失败', className: 'bg-rose-500/20 text-rose-400' },
  cancelled: { text: '已取消', className: 'bg-amber-500/20 text-amber-400' },
};

const TYPE_LABELS: Record<string, string> = {
  engine_comparison: '引擎对比',
  component_ablation: '组件消融',
  threshold_sweep: '阈值扫描',
  scenario_analysis: '场景分析',
  full_suite: '全量实验',
};

interface AblationPanelProps {
  onSelectExperiment?: (expId: string) => void;
}

export function AblationPanel({ onSelectExperiment }: AblationPanelProps) {
  const { recordings } = useRecordings();
  const { experiments, loading: expLoading, refresh: refreshExps } = useExperiments();
  const { startExperiment, stopExperiment, deleteExperiment, startFullSuite, starting } = useExperimentControl();
  const wsStatus = useDataLabWebSocket();

  const [selectedRecording, setSelectedRecording] = useState('');
  const [experimentType, setExperimentType] = useState<ExperimentType>('engine_comparison');
  const [selectedEngines, setSelectedEngines] = useState<string[]>(
    ENGINE_OPTIONS.map((e) => e.key)
  );

  // 全量实验正负样本选择（支持多选）
  const [selectedPositiveIds, setSelectedPositiveIds] = useState<string[]>([]);
  const [selectedNegativeIds, setSelectedNegativeIds] = useState<string[]>([]);

  // 多选实验
  const [selectedExpIds, setSelectedExpIds] = useState<Set<string>>(new Set());

  const toggleEngine = (key: string) => {
    setSelectedEngines((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const handleStart = async () => {
    if (!selectedRecording) return;
    await startExperiment(
      selectedRecording,
      experimentType,
      experimentType === 'engine_comparison' || experimentType === 'scenario_analysis'
        ? selectedEngines
        : undefined,
    );
    refreshExps();
  };

  const togglePositive = (id: string) => {
    setSelectedPositiveIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const toggleNegative = (id: string) => {
    setSelectedNegativeIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const handleFullSuite = async () => {
    if (selectedPositiveIds.length === 0 || selectedNegativeIds.length === 0) return;
    await startFullSuite(selectedPositiveIds, selectedNegativeIds);
    refreshExps();
  };

  const handleStop = async (exp: AblationExperiment) => {
    await stopExperiment(exp.id);
    refreshExps();
  };

  const handleDelete = async (exp: AblationExperiment) => {
    if (!confirm(`确定删除实验 ${exp.id}？`)) return;
    try {
      await deleteExperiment(exp.id);
      refreshExps();
    } catch (e: any) {
      alert('删除失败: ' + (e.message || '未知错误'));
    }
  };

  const toggleSelectExp = (expId: string) => {
    setSelectedExpIds((prev) => {
      const next = new Set(prev);
      if (next.has(expId)) {
        next.delete(expId);
      } else {
        next.add(expId);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedExpIds.size === experiments.length && experiments.length > 0) {
      setSelectedExpIds(new Set());
    } else {
      setSelectedExpIds(new Set(experiments.map((e) => e.id)));
    }
  };

  const handleBatchDelete = async () => {
    if (selectedExpIds.size === 0) return;
    if (!confirm(`确定删除选中的 ${selectedExpIds.size} 个实验？`)) return;
    for (const expId of selectedExpIds) {
      try {
        await deleteExperiment(expId);
      } catch {
        // ignore individual failures
      }
    }
    setSelectedExpIds(new Set());
    refreshExps();
  };

  const runningExp = wsStatus?.experiment;

  const showEngineSelector =
    experimentType === 'engine_comparison' || experimentType === 'scenario_analysis';

  return (
    <div className="space-y-6">
      {/* 控制栏 */}
      <div className="flex flex-wrap items-start gap-4 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center gap-2">
          <Beaker className="h-5 w-5 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">消融实验</span>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-500">选择录制素材</label>
          <select
            value={selectedRecording}
            onChange={(e) => setSelectedRecording(e.target.value)}
            disabled={starting || !!runningExp}
            className="min-w-[200px] rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-sky-500"
          >
            <option value="">-- 请选择 --</option>
            {recordings.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} ({r.camera_id}, {r.frame_count}帧)
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-500">实验类型</label>
          <select
            value={experimentType}
            onChange={(e) => setExperimentType(e.target.value as ExperimentType)}
            disabled={starting || !!runningExp}
            className="min-w-[220px] rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-sky-500"
          >
            {EXPERIMENT_TYPE_OPTIONS.map((opt) => (
              <option key={opt.key} value={opt.key}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="max-w-[260px] text-xs text-slate-500">
            {EXPERIMENT_TYPE_OPTIONS.find((o) => o.key === experimentType)?.desc}
          </p>
        </div>

        {showEngineSelector && (
          <div className="flex flex-col gap-2">
            <label className="text-xs text-slate-500">选择引擎</label>
            <div className="flex flex-wrap gap-2">
              {ENGINE_OPTIONS.map((opt) => (
                <label
                  key={opt.key}
                  className={`flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors ${
                    selectedEngines.includes(opt.key)
                      ? 'border-sky-500/50 bg-sky-500/10 text-sky-300'
                      : 'border-slate-700 bg-slate-800 text-slate-400'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedEngines.includes(opt.key)}
                    onChange={() => toggleEngine(opt.key)}
                    disabled={starting || !!runningExp}
                    className="h-3.5 w-3.5 rounded border-slate-600"
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>
        )}

        {!runningExp ? (
          <>
            <button
              onClick={handleStart}
              disabled={!selectedRecording || (showEngineSelector && selectedEngines.length === 0) || starting}
              className="flex items-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {starting ? '启动中...' : '开始实验'}
            </button>

            {/* 全量实验：支持多选正负样本 */}
            <div className="flex items-start gap-3">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-slate-400">
                  正样本（含 waving）— 已选 {selectedPositiveIds.length}
                </label>
                <div className="max-h-[120px] min-w-[180px] overflow-y-auto rounded-lg border border-slate-700 bg-slate-800 p-2">
                  {recordings.filter((r) => r.manual_label === 'positive').length === 0 ? (
                    <span className="text-xs text-slate-500">无正样本</span>
                  ) : (
                    recordings
                      .filter((r) => r.manual_label === 'positive')
                      .map((r) => (
                        <label
                          key={r.id}
                          className="flex cursor-pointer items-center gap-1.5 py-0.5 text-xs text-slate-200 hover:text-white"
                        >
                          <input
                            type="checkbox"
                            checked={selectedPositiveIds.includes(r.id)}
                            onChange={() => togglePositive(r.id)}
                            disabled={starting || !!runningExp}
                            className="h-3.5 w-3.5 rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-sky-500"
                          />
                          <span className="truncate">{r.id} ({r.frame_count}帧)</span>
                        </label>
                      ))
                  )}
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-medium text-slate-400">
                  负样本（不含 waving）— 已选 {selectedNegativeIds.length}
                </label>
                <div className="max-h-[120px] min-w-[180px] overflow-y-auto rounded-lg border border-slate-700 bg-slate-800 p-2">
                  {recordings.filter((r) => r.manual_label === 'negative').length === 0 ? (
                    <span className="text-xs text-slate-500">无负样本</span>
                  ) : (
                    recordings
                      .filter((r) => r.manual_label === 'negative')
                      .map((r) => (
                        <label
                          key={r.id}
                          className="flex cursor-pointer items-center gap-1.5 py-0.5 text-xs text-slate-200 hover:text-white"
                        >
                          <input
                            type="checkbox"
                            checked={selectedNegativeIds.includes(r.id)}
                            onChange={() => toggleNegative(r.id)}
                            disabled={starting || !!runningExp}
                            className="h-3.5 w-3.5 rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-sky-500"
                          />
                          <span className="truncate">{r.id} ({r.frame_count}帧)</span>
                        </label>
                      ))
                  )}
                </div>
              </div>
              <button
                onClick={handleFullSuite}
                disabled={selectedPositiveIds.length === 0 || selectedNegativeIds.length === 0 || starting}
                className="mt-5 flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
                title="自动依次运行：引擎对比 → 组件消融 → 阈值扫描 → 场景分析（正负样本分别评估）"
              >
                {starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
                一键全量实验
              </button>
            </div>
          </>
        ) : (
          <button
            onClick={() => handleStop(runningExp)}
            className="flex items-center gap-2 rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600"
          >
            <Square className="h-4 w-4 fill-current" />
            停止实验
          </button>
        )}

        <button
          onClick={refreshExps}
          disabled={expLoading}
          className="ml-auto flex items-center gap-1 rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-400 hover:bg-slate-800"
        >
          <RefreshCw className={`h-4 w-4 ${expLoading ? 'animate-spin' : ''}`} />
          刷新
        </button>
      </div>

      {/* 运行中实验进度 */}
      {runningExp && (
        <div className="rounded-xl border border-sky-800/50 bg-sky-900/10 p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium text-sky-300">
              实验运行中: {runningExp.id}
              {runningExp.experiment_type && (
                <span className="ml-2 text-xs text-sky-400">
                  ({runningExp.experiment_type})
                </span>
              )}
            </span>
            <span className="text-xs text-sky-400">
              {runningExp.current_frame} / {runningExp.total_frames} 帧
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-sky-500 transition-all"
              style={{ width: `${(runningExp.progress * 100).toFixed(1)}%` }}
            />
          </div>
          <div className="mt-1 text-xs text-sky-400">
            进度: {(runningExp.progress * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* 实验列表 */}
      <div className="overflow-hidden rounded-xl border border-slate-800">
        <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/80 px-4 py-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={experiments.length > 0 && selectedExpIds.size === experiments.length}
              onChange={toggleSelectAll}
              className="h-4 w-4 rounded border-slate-600"
            />
            <span className="text-xs text-slate-400">
              已选 {selectedExpIds.size} / {experiments.length}
            </span>
          </div>
          {selectedExpIds.size > 0 && (
            <button
              onClick={handleBatchDelete}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-rose-400 hover:bg-rose-500/10"
            >
              <Trash2 className="h-3.5 w-3.5" />
              批量删除
            </button>
          )}
        </div>
        <table className="w-full text-sm">
          <thead className="bg-slate-900/80 text-left text-slate-400">
            <tr>
              <th className="px-2 py-3 font-medium"></th>
              <th className="px-4 py-3 font-medium">ID</th>
              <th className="px-4 py-3 font-medium">类型</th>
              <th className="px-4 py-3 font-medium">录制素材</th>
              <th className="px-4 py-3 font-medium">引擎</th>
              <th className="px-4 py-3 font-medium">状态</th>
              <th className="px-4 py-3 font-medium">进度</th>
              <th className="px-4 py-3 font-medium text-right">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {experiments.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                  暂无实验记录
                </td>
              </tr>
            )}
            {experiments.map((exp) => (
              <tr key={exp.id} className="hover:bg-slate-800/50">
                <td className="px-2 py-3">
                  <input
                    type="checkbox"
                    checked={selectedExpIds.has(exp.id)}
                    onChange={() => toggleSelectExp(exp.id)}
                    className="h-4 w-4 rounded border-slate-600"
                  />
                </td>
                <td className="px-4 py-3 font-mono text-slate-300">{exp.id}</td>
                <td className="px-4 py-3 text-slate-400">
                  <span className="rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 text-xs">
                    {TYPE_LABELS[exp.experiment_type ?? 'engine_comparison'] ?? exp.experiment_type}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-400">{exp.recording_id}</td>
                <td className="px-4 py-3 text-slate-400">
                  <div className="flex flex-wrap gap-1">
                    {exp.engine_names.slice(0, 3).map((n) => (
                      <span
                        key={n}
                        className="rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 text-xs"
                      >
                        {n}
                      </span>
                    ))}
                    {exp.engine_names.length > 3 && (
                      <span className="text-xs text-slate-500">+{exp.engine_names.length - 3}</span>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span
                    className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs ${
                      STATUS_LABELS[exp.status]?.className ?? STATUS_LABELS.pending.className
                    }`}
                  >
                    {exp.status === 'completed' && <CheckCircle2 className="h-3 w-3" />}
                    {exp.status === 'failed' && <XCircle className="h-3 w-3" />}
                    {exp.status === 'running' && <Loader2 className="h-3 w-3 animate-spin" />}
                    {STATUS_LABELS[exp.status]?.text ?? exp.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {(exp.progress * 100).toFixed(0)}%
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-2">
                    {exp.status === 'completed' && (
                      <button
                        onClick={() => onSelectExperiment?.(exp.id)}
                        className="rounded px-2 py-1 text-xs text-sky-400 hover:bg-sky-500/10"
                      >
                        查看分析
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(exp)}
                      className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-rose-400 hover:bg-rose-500/10"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
