import { useMemo, useState } from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  FileJson,
  FileSpreadsheet,
  FileText,
  RefreshCw,
  Award,
  Zap,
  Shield,
  Activity,
  Layers,
  Clock,
  MapPin,
  TrendingUp,
  Image,
  FileDown,
} from 'lucide-react';
import { useAnalysisReport, useExperimentControl } from './useDataLab';
import { HighlightCard } from './HighlightCard';
import type { EngineStats, AnalysisReport } from './types';

interface AnalysisDashboardProps {
  experimentId?: string;
  onRequestNew?: () => void;
}

const ENGINE_COLORS: Record<string, string> = {
  simple: '#94a3b8',
  simple_minicpm: '#34d399',
  minicpm: '#60a5fa',
};

const ENGINE_LABELS: Record<string, string> = {
  simple: 'Simple',
  simple_minicpm: 'Simple+MiniCPM ★',
  minicpm: 'MiniCPM',
};

const TAB_OPTIONS = [
  { key: 'overview' as const, label: '总览' },
  { key: 'engine' as const, label: '引擎对比' },
  { key: 'component' as const, label: '组件消融' },
  { key: 'threshold' as const, label: '阈值扫描' },
  { key: 'scenario' as const, label: '场景分析' },
];

export function AnalysisDashboard({ experimentId, onRequestNew }: AnalysisDashboardProps) {
  const { report, loading, error, refresh } = useAnalysisReport(experimentId);
  const [activeTab, setActiveTab] = useState<(typeof TAB_OPTIONS)[number]['key']>('overview');

  if (!experimentId) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50">
        <div className="text-center text-slate-500">
          <Activity className="mx-auto mb-2 h-8 w-8" />
          <p>请在"消融实验"面板选择一个已完成的实验查看分析结果</p>
          <button
            onClick={onRequestNew}
            className="mt-3 rounded-lg bg-sky-600 px-4 py-2 text-sm text-white hover:bg-sky-500"
          >
            前往实验面板
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500">
        <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
        正在生成分析报告...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-800/50 bg-red-900/20 px-4 py-3 text-sm text-red-300">
        加载失败: {error}
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500">
        暂无分析报告
      </div>
    );
  }

  const isFullSuite = report.experiment_type === 'full_suite';

  return (
    <div className="space-y-6">
      {isFullSuite && (
        <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
          {TAB_OPTIONS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'bg-sky-500/10 text-sky-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}

      {/* 总览 */}
      {(!isFullSuite || activeTab === 'overview') && (
        <>
          <SummaryCards report={report} />
          <ConclusionSection report={report} experimentId={experimentId} onRefresh={refresh} />
        </>
      )}

      {/* 引擎对比 */}
      {(!isFullSuite || activeTab === 'engine') && report.engine_stats.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <DetectionRateChart stats={report.engine_stats} />
            <CalibrationScoreChart stats={report.engine_stats} calibrationScores={report.calibration_scores} />
            <RadarComparisonChart stats={report.engine_stats} prf1={report.precision_recall_f1} calibrationScores={report.calibration_scores} />
            <AgreementHeatmap matrix={report.agreement_matrix} />
          </div>
          {report.temporal_metrics.length > 0 && (
            <TemporalMetricsTable metrics={report.temporal_metrics} />
          )}
          <StatsTable stats={report.engine_stats} prf1={report.precision_recall_f1} calibrationScores={report.calibration_scores} />
        </>
      )}

      {/* 组件消融 */}
      {(!isFullSuite || activeTab === 'component') && report.component_contributions.length > 0 && (
        <>
          <ComponentContributionChart contributions={report.component_contributions} />
          <ComponentDetailList contributions={report.component_contributions} />
        </>
      )}

      {/* 阈值扫描 */}
      {(!isFullSuite || activeTab === 'threshold') && report.pr_curve.length > 0 && (
        <PRCurveChart prCurve={report.pr_curve} />
      )}

      {/* 场景分析 */}
      {(!isFullSuite || activeTab === 'scenario') && report.scenario_stats.length > 0 && (
        <ScenarioComparisonChart scenarioStats={report.scenario_stats} />
      )}
    </div>
  );
}

// ------------------------------------------------------------------
// Summary Cards
// ------------------------------------------------------------------

function SummaryCards({ report }: { report: AnalysisReport }) {
  const smc = report.engine_stats.find((s) => s.engine_name === 'simple_minicpm');
  const adv = report.simple_minicpm_advantage;

  let bestF1 = -1;
  let bestF1Engine = '';
  Object.entries(report.precision_recall_f1).forEach(([name, vals]) => {
    if (vals.f1 > bestF1) {
      bestF1 = vals.f1;
      bestF1Engine = name;
    }
  });

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <HighlightCard
        title="总帧数"
        value={report.total_frames}
        unit="帧"
        description={`共识基线: ${report.consensus_baseline_frames} 帧`}
        color="blue"
      />
      {smc && (
        <HighlightCard
          title="Simple+MiniCPM 检测率"
          value={(smc.detection_rate * 100).toFixed(1)}
          unit="%"
          description={`${smc.waving_frames} 帧 waving / ${smc.positive_segments} 片段`}
          color="green"
        />
      )}
      {bestF1 >= 0 && (
        <HighlightCard
          title="最佳 F1 得分"
          value={bestF1.toFixed(3)}
          description={`${ENGINE_LABELS[bestF1Engine] || bestF1Engine}`}
          color="blue"
        />
      )}
      {adv.overall_score > 0 && (
        <HighlightCard
          title="综合优势得分"
          value={adv.overall_score.toFixed(1)}
          description="精度、召回、鲁棒性加权"
          color="amber"
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------------
// Detection Rate Bar Chart
// ------------------------------------------------------------------

function DetectionRateChart({ stats }: { stats: EngineStats[] }) {
  const data = useMemo(
    () =>
      stats.map((s) => ({
        name: ENGINE_LABELS[s.engine_name] || s.engine_name,
        rate: +(s.detection_rate * 100).toFixed(1),
        fill: ENGINE_COLORS[s.engine_name] || '#94a3b8',
      })),
    [stats]
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Activity className="h-4 w-4 text-sky-400" />
        检测率对比
      </h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
          />
          <Bar dataKey="rate" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------
// Confidence Distribution
// ------------------------------------------------------------------

function CalibrationScoreChart({
  stats,
  calibrationScores,
}: {
  stats: EngineStats[];
  calibrationScores?: Record<string, number>;
}) {
  const data = useMemo(
    () =>
      stats.map((s) => ({
        name: ENGINE_LABELS[s.engine_name] || s.engine_name,
        score: +(Math.max(0, calibrationScores?.[s.engine_name] ?? 0) * 100).toFixed(1),
        fill: ENGINE_COLORS[s.engine_name] || '#94a3b8',
      })),
    [stats, calibrationScores]
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Zap className="h-4 w-4 text-amber-400" />
        置信度校准度
      </h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} unit="%" />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, '校准度']}
          />
          <Bar dataKey="score" fill="#34d399" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-slate-500">
        越接近 100% 表示模型置信度与真实精度越一致
      </p>
    </div>
  );
}

// ------------------------------------------------------------------
// Radar Chart
// ------------------------------------------------------------------

function RadarComparisonChart({ stats, prf1, calibrationScores }: { stats: EngineStats[]; prf1?: Record<string, { precision?: number }>; calibrationScores?: Record<string, number> }) {
  const dimensions = [
    { key: 'detection_rate', label: '检测率', max: 1 },
    { key: 'calibration', label: '置信度校准度', max: 1 },
    { key: 'robustness', label: '鲁棒性', max: 1 },
    { key: 'efficiency', label: '资源效率', max: 1 },
    { key: 'precision', label: '精度', max: 1 },
  ];

  const data = useMemo(() => {
    const maxDetection = Math.max(...stats.map((s) => s.detection_rate), 0.001);
    const maxInference = Math.max(...stats.map((s) => s.inference_count), 0.001);
    const maxFp = Math.max(...stats.map((s) => s.false_positive_estimate), 0.001);
    const maxCalib = Math.max(...Object.values(calibrationScores || {}), 0.001);

    return dimensions.map((dim) => {
      const row: Record<string, any> = { dimension: dim.label };
      for (const s of stats) {
        let val = 0;
        if (dim.key === 'detection_rate') val = s.detection_rate / maxDetection;
        else if (dim.key === 'calibration') val = (calibrationScores?.[s.engine_name] || 0) / maxCalib;
        else if (dim.key === 'robustness')
          val = Math.max(0, 1 - s.noise_rejection_rate);
        else if (dim.key === 'efficiency') val = Math.max(0, 1 - s.inference_count / maxInference);
        else if (dim.key === 'precision') {
          // 优先使用跨正负样本计算的 precision（TP / (TP + FP)）
          const crossPrecision = prf1?.[s.engine_name]?.precision;
          if (crossPrecision !== undefined && crossPrecision >= 0) {
            val = crossPrecision;
          } else {
            val = Math.max(0, 1 - s.false_positive_estimate / maxFp);
          }
        }
        row[s.engine_name] = Math.min(1, Math.max(0, val));
      }
      return row;
    });
  }, [stats, prf1, calibrationScores]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Award className="h-4 w-4 text-emerald-400" />
        多维度能力雷达图
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <RadarChart data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
          {stats.map((s) => (
            <Radar
              key={s.engine_name}
              name={ENGINE_LABELS[s.engine_name] || s.engine_name}
              dataKey={s.engine_name}
              stroke={ENGINE_COLORS[s.engine_name] || '#94a3b8'}
              fill={ENGINE_COLORS[s.engine_name] || '#94a3b8'}
              fillOpacity={s.engine_name === 'simple_minicpm' ? 0.35 : 0.05}
              strokeWidth={s.engine_name === 'simple_minicpm' ? 3 : 1.5}
            />
          ))}
          <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------
// Agreement Heatmap
// ------------------------------------------------------------------

function AgreementHeatmap({ matrix }: { matrix: AnalysisReport['agreement_matrix'] }) {
  const names = matrix.engine_names;
  if (names.length === 0) return null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Shield className="h-4 w-4 text-purple-400" />
        引擎一致率矩阵
      </h3>
      <div className="overflow-x-auto">
        <div className="inline-block">
          <div className="flex">
            <div className="w-24" />
            {names.map((n) => (
              <div
                key={n}
                className="flex w-20 items-center justify-center py-2 text-xs text-slate-400"
              >
                {ENGINE_LABELS[n] || n}
              </div>
            ))}
          </div>
          {names.map((rowName) => (
            <div key={rowName} className="flex">
              <div className="flex w-24 items-center justify-end pr-3 text-xs text-slate-400">
                {ENGINE_LABELS[rowName] || rowName}
              </div>
              {names.map((colName) => {
                const val = matrix.matrix[rowName]?.[colName] ?? 0;
                const isDiag = rowName === colName;
                return (
                  <div
                    key={colName}
                    className={`flex h-10 w-20 items-center justify-center text-xs font-medium ${
                      isDiag ? 'text-emerald-400' : 'text-slate-200'
                    }`}
                    style={{
                      backgroundColor: isDiag
                        ? 'rgba(16, 185, 129, 0.15)'
                        : `rgba(56, 189, 248, ${val * 0.4})`,
                    }}
                    title={`${ENGINE_LABELS[rowName]} vs ${ENGINE_LABELS[colName]}: ${(val * 100).toFixed(1)}%`}
                  >
                    {(val * 100).toFixed(0)}%
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------
// PR Curve Chart
// ------------------------------------------------------------------

function PRCurveChart({ prCurve }: { prCurve: AnalysisReport['pr_curve'] }) {
  const { prSeries, rocSeries, prBounds, rocBounds } = useMemo(() => {
    const prS: Record<string, { x: number; y: number }[]> = {};
    const rocS: Record<string, { x: number; y: number }[]> = {};
    for (const p of prCurve) {
      const en = p.engine_name;
      if (!prS[en]) prS[en] = [];
      if (!rocS[en]) rocS[en] = [];
      prS[en].push({ x: p.recall, y: p.precision });
      rocS[en].push({ x: p.fpr, y: p.tpr });
    }
    const allPrX = prCurve.map((p) => p.recall);
    const allRocX = prCurve.map((p) => p.fpr);
    const pad = (min: number, max: number) => {
      const r = max - min || 0.05;
      return { min: Math.max(0, min - r * 0.05), max: Math.min(1, max + r * 0.05) };
    };
    return {
      prSeries: prS,
      rocSeries: rocS,
      prBounds: pad(Math.min(...allPrX, 0), Math.max(...allPrX, 1)),
      rocBounds: pad(Math.min(...allRocX, 0), Math.max(...allRocX, 1)),
    };
  }, [prCurve]);

  const engineOrder = useMemo(() => Object.keys(prSeries), [prSeries]);

  const drawChart = (
    series: Record<string, { x: number; y: number }[]>,
    bounds: { min: number; max: number },
    yBounds: { min: number; max: number },
    xLabel: string,
    yLabel: string,
    title: string
  ) => {
    const w = 600;
    const h = 320;
    const m = { t: 30, r: 20, b: 45, l: 50 };
    const cw = w - m.l - m.r;
    const ch = h - m.t - m.b;

    const tx = (x: number) => m.l + ((x - bounds.min) / (bounds.max - bounds.min)) * cw;
    const ty = (y: number) => m.t + ch - ((y - yBounds.min) / (yBounds.max - yBounds.min)) * ch;

    const gridLines = [];
    for (let i = 0; i <= 5; i++) {
      const x = m.l + (cw * i) / 5;
      const y = m.t + (ch * i) / 5;
      const xv = bounds.min + ((bounds.max - bounds.min) * i) / 5;
      const yv = yBounds.max - ((yBounds.max - yBounds.min) * i) / 5;
      gridLines.push(
        <line key={`gx${i}`} x1={x} y1={m.t} x2={x} y2={m.t + ch} stroke="#1e293b" strokeWidth={1} strokeDasharray="4,4" />,
        <text key={`gxt${i}`} x={x} y={m.t + ch + 16} textAnchor="middle" fill="#64748b" fontSize={10}>{xv.toFixed(2)}</text>,
        <line key={`gy${i}`} x1={m.l} y1={y} x2={m.l + cw} y2={y} stroke="#1e293b" strokeWidth={1} />,
        <text key={`gyt${i}`} x={m.l - 8} y={y + 4} textAnchor="end" fill="#64748b" fontSize={10}>{yv.toFixed(2)}</text>
      );
    }

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: h }}>
        <rect width={w} height={h} fill="#0f172a" rx={8} />
        <text x={w / 2} y={22} textAnchor="middle" fill="#e2e8f0" fontSize={14} fontWeight={600}>{title}</text>
        {gridLines}
        <text x={m.l + cw / 2} y={h - 4} textAnchor="middle" fill="#94a3b8" fontSize={12}>{xLabel}</text>
        <text x={16} y={m.t + ch / 2} textAnchor="middle" fill="#94a3b8" fontSize={12} transform={`rotate(-90, 16, ${m.t + ch / 2})`}>{yLabel}</text>
        {engineOrder.map((eng) => {
          const pts = series[eng] || [];
          if (pts.length === 0) return null;
          const sorted = [...pts].sort((a, b) => a.x - b.x);
          const color = ENGINE_COLORS[eng] || '#38bdf8';
          const ptsStr = sorted.map((p) => `${tx(p.x)},${ty(p.y)}`).join(' ');
          return (
            <g key={eng}>
              <polyline points={ptsStr} fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
              {/* 仅在首尾画点，避免密集重叠 */}
              {sorted.length > 0 && (
                <>
                  <circle cx={tx(sorted[0].x)} cy={ty(sorted[0].y)} r={3} fill={color} stroke="#0f172a" strokeWidth={1.5} />
                  <circle cx={tx(sorted[sorted.length - 1].x)} cy={ty(sorted[sorted.length - 1].y)} r={3} fill={color} stroke="#0f172a" strokeWidth={1.5} />
                </>
              )}
            </g>
          );
        })}
        <g transform={`translate(${m.l + cw + 4}, ${m.t})`}>
          {engineOrder.map((eng, i) => {
            const color = ENGINE_COLORS[eng] || '#38bdf8';
            return (
              <g key={eng} transform={`translate(0, ${i * 18})`}>
                <rect x={0} y={-5} width={14} height={3} rx={1.5} fill={color} />
                <text x={20} y={0} fill="#94a3b8" fontSize={11}>{ENGINE_LABELS[eng] || eng}</text>
              </g>
            );
          })}
        </g>
      </svg>
    );
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:col-span-2">
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
          <TrendingUp className="h-4 w-4 text-sky-400" />
          PR 曲线（Precision-Recall）
        </h3>
        {drawChart(prSeries, prBounds, prBounds, 'Recall', 'Precision', 'PR 曲线')}
      </div>
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
          <TrendingUp className="h-4 w-4 text-sky-400" />
          ROC 曲线（TPR-FPR）
        </h3>
        {drawChart(rocSeries, rocBounds, { min: 0, max: 1 }, 'FPR', 'TPR', 'ROC 曲线')}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------
// Component Contribution Chart
// ------------------------------------------------------------------

function ComponentContributionChart({
  contributions,
}: {
  contributions: AnalysisReport['component_contributions'];
}) {
  const data = useMemo(
    () =>
      contributions.map((c) => ({
        name: c.component_name,
        score: +c.contribution_score.toFixed(1),
        fill: c.contribution_score > 0 ? '#34d399' : '#f87171',
      })),
    [contributions]
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 lg:col-span-2">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Layers className="h-4 w-4 text-emerald-400" />
        组件消融贡献分析（正数 = 该组件提升性能）
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={80} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, '贡献分']}
          />
          <Bar dataKey="score" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------
// Component Detail List (with Chinese descriptions)
// ------------------------------------------------------------------

function ComponentDetailList({
  contributions,
}: {
  contributions: AnalysisReport['component_contributions'];
}) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-300">组件详细说明</h3>
      {contributions.map((c) => {
        const isPositive = c.contribution_score > 0;
        return (
          <div
            key={c.component_name}
            className="flex gap-3 rounded-xl border border-slate-800 bg-slate-900/50 p-4"
          >
            <div
              className={`w-1 shrink-0 self-stretch rounded-full ${
                isPositive ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-slate-200">{c.component_name}</h4>
                <span
                  className={`rounded px-2 py-0.5 text-xs font-medium ${
                    isPositive
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-rose-500/10 text-rose-400'
                  }`}
                >
                  {isPositive ? '+' : ''}
                  {c.contribution_score.toFixed(1)}%
                </span>
              </div>
              <p className="mt-1 text-sm leading-relaxed text-slate-400">
                {c.component_description}
              </p>
              <div className="mt-2 flex gap-4 text-xs text-slate-500">
                <span>完整 F1: {c.full_f1.toFixed(3)}</span>
                <span>消融后 F1: {c.ablated_f1.toFixed(3)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------
// Scenario Comparison Chart
// ------------------------------------------------------------------

function ScenarioComparisonChart({
  scenarioStats,
}: {
  scenarioStats: AnalysisReport['scenario_stats'];
}) {
  const byType = useMemo(() => {
    const groups: Record<string, typeof scenarioStats> = {};
    for (const s of scenarioStats) {
      if (!groups[s.scenario_type]) groups[s.scenario_type] = [];
      groups[s.scenario_type].push(s);
    }
    return groups;
  }, [scenarioStats]);

  return (
    <div className="space-y-4 lg:col-span-2">
      {Object.entries(byType).map(([type, scenarios]) => {
        const data = scenarios.map((s) => {
          const row: Record<string, any> = { name: s.scenario_name };
          Object.entries(s.engine_results).forEach(([eng, vals]) => {
            row[eng] = +(vals.f1 * 100).toFixed(1);
          });
          return row;
        });

        const engines = Object.keys(scenarios[0]?.engine_results || {});

        return (
          <div key={type} className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
              <MapPin className="h-4 w-4 text-amber-400" />
              场景分析 — {type === 'velocity' ? '速度' : type === 'distance' ? '距离' : type === 'hand' ? '左右手' : type}
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'F1']}
                />
                <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                {engines.map((eng) => (
                  <Bar
                    key={eng}
                    dataKey={eng}
                    name={ENGINE_LABELS[eng] || eng}
                    fill={ENGINE_COLORS[eng] || '#94a3b8'}
                    radius={[4, 4, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------
// Temporal Metrics Table
// ------------------------------------------------------------------

function TemporalMetricsTable({
  metrics,
}: {
  metrics: AnalysisReport['temporal_metrics'];
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 lg:col-span-2">
      <h3 className="flex items-center gap-2 bg-slate-900/80 px-4 py-3 text-sm font-semibold text-slate-300">
        <Clock className="h-4 w-4 text-sky-400" />
        时序一致性指标
      </h3>
      <table className="w-full text-sm">
        <thead className="bg-slate-900/60 text-left text-slate-400">
          <tr>
            <th className="px-4 py-2 font-medium">引擎</th>
            <th className="px-4 py-2 font-medium">响应延迟</th>
            <th className="px-4 py-2 font-medium">碎片化率</th>
            <th className="px-4 py-2 font-medium">平均片段长度</th>
            <th className="px-4 py-2 font-medium">稳定性 CV</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {metrics.map((m) => (
            <tr key={m.engine_name} className="hover:bg-slate-800/50">
              <td className="px-4 py-2 font-medium text-slate-300">
                {ENGINE_LABELS[m.engine_name] || m.engine_name}
              </td>
              <td className="px-4 py-2 text-slate-400">
                {m.response_latency_mean.toFixed(1)}±{m.response_latency_std.toFixed(1)} 帧
              </td>
              <td className="px-4 py-2 text-slate-400">{m.fragmentation_rate.toFixed(1)} 次/s</td>
              <td className="px-4 py-2 text-slate-400">{m.avg_positive_duration.toFixed(1)} 帧</td>
              <td className="px-4 py-2 text-slate-400">{m.detection_stability_cv.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------
// Conclusion & Export
// ------------------------------------------------------------------

function ConclusionSection({
  report,
  experimentId,
  onRefresh,
}: {
  report: AnalysisReport;
  experimentId: string;
  onRefresh: () => void;
}) {
  const { exportMarkdown, exportCharts, exportPdf } = useExperimentControl();

  const handleExportCsv = async () => {
    window.open(`/api/datalab/experiments/${experimentId}/export/csv`, '_blank');
  };

  const handleExportJson = async () => {
    window.open(`/api/datalab/experiments/${experimentId}/export/json`, '_blank');
  };

  const handleExportMd = async () => {
    try {
      await exportMarkdown(experimentId);
    } catch (e: any) {
      alert('导出失败: ' + (e.message || '未知错误'));
    }
  };

  const handleExportCharts = async () => {
    try {
      await exportCharts(experimentId);
    } catch (e: any) {
      alert('导出失败: ' + (e.message || '未知错误'));
    }
  };

  const handleExportPdf = async () => {
    try {
      await exportPdf(experimentId);
    } catch (e: any) {
      alert('导出失败: ' + (e.message || '未知错误'));
    }
  };

  return (
    <div className="space-y-4">
      {report.simple_minicpm_advantage.overall_score > 0 && (
        <div className="rounded-xl border border-emerald-800/40 bg-emerald-900/10 p-5">
          <h3 className="mb-3 flex items-center gap-2 text-base font-semibold text-emerald-400">
            <Award className="h-5 w-5" />
            Simple+MiniCPM 优势分析
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
            <MetricItem
              label="vs Simple 精度提升"
              value={`+${report.simple_minicpm_advantage.vs_simple_precision_gain.toFixed(1)}%`}
            />
            <MetricItem
              label="vs MiniCPM 召回提升"
              value={`+${report.simple_minicpm_advantage.vs_minicpm_recall_gain.toFixed(1)}%`}
            />
            <MetricItem
              label="静止鲁棒性提升"
              value={`${report.simple_minicpm_advantage.noise_rejection_score.toFixed(1)}%`}
            />
            <MetricItem
              label="推理效率增益"
              value={`${report.simple_minicpm_advantage.latency_efficiency_gain.toFixed(1)}%`}
            />
          </div>
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <h3 className="mb-3 text-sm font-semibold text-slate-300">实验结论</h3>
        <div className="prose prose-invert prose-sm max-w-none text-slate-300">
          {report.conclusion_markdown.split('\n').map((line, i) => {
            if (line.startsWith('## ')) {
              return (
                <h2 key={i} className="mt-4 text-lg font-bold text-slate-200">
                  {line.replace('## ', '')}
                </h2>
              );
            }
            if (line.startsWith('### ')) {
              return (
                <h3 key={i} className="mt-3 text-base font-semibold text-slate-200">
                  {line.replace('### ', '')}
                </h3>
              );
            }
            if (line.startsWith('- ')) {
              return (
                <li key={i} className="ml-4 text-slate-400">
                  {line.replace('- ', '')}
                </li>
              );
            }
            if (line.startsWith('> ')) {
              return (
                <blockquote
                  key={i}
                  className="mt-2 border-l-4 border-emerald-500/50 bg-emerald-500/5 pl-4 text-emerald-300"
                >
                  {line.replace('> ', '')}
                </blockquote>
              );
            }
            if (line.trim() === '') {
              return <div key={i} className="h-2" />;
            }
            return (
              <p key={i} className="text-slate-400">
                {line}
              </p>
            );
          })}
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleExportCsv}
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <FileSpreadsheet className="h-4 w-4" />
          导出 CSV
        </button>
        <button
          onClick={handleExportJson}
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <FileJson className="h-4 w-4" />
          导出 JSON
        </button>
        <button
          onClick={handleExportMd}
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <FileText className="h-4 w-4" />
          导出 Markdown
        </button>
        <button
          onClick={handleExportCharts}
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <Image className="h-4 w-4" />
          导出图表
        </button>
        <button
          onClick={handleExportPdf}
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <FileDown className="h-4 w-4" />
          导出 PDF
        </button>
        <button
          onClick={onRefresh}
          className="ml-auto flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <RefreshCw className="h-4 w-4" />
          重新生成报告
        </button>
      </div>
    </div>
  );
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-900/60 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-bold text-emerald-400">{value}</p>
    </div>
  );
}

// ------------------------------------------------------------------
// Stats Table (enhanced with P/R/F1)
// ------------------------------------------------------------------

function StatsTable({
  stats,
  prf1,
  calibrationScores,
}: {
  stats: EngineStats[];
  prf1: AnalysisReport['precision_recall_f1'];
  calibrationScores?: Record<string, number>;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-sm">
        <thead className="bg-slate-900/80 text-left text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">引擎</th>
            <th className="px-4 py-3 font-medium">检测率</th>
            <th className="px-4 py-3 font-medium">Precision</th>
            <th className="px-4 py-3 font-medium">Recall</th>
            <th className="px-4 py-3 font-medium">F1</th>
            <th className="px-4 py-3 font-medium">校准度</th>
            <th className="px-4 py-3 font-medium">平均耗时</th>
            <th className="px-4 py-3 font-medium">连续片段</th>
            <th className="px-4 py-3 font-medium">估算误检率</th>
            <th className="px-4 py-3 font-medium">静止误检率</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {stats.map((s) => {
            const isSmc = s.engine_name === 'simple_minicpm';
            const f1Data = prf1[s.engine_name];
            const calib = calibrationScores?.[s.engine_name] ?? 0;
            return (
              <tr
                key={s.engine_name}
                className={`${isSmc ? 'bg-emerald-500/5' : 'hover:bg-slate-800/50'}`}
              >
                <td
                  className={`px-4 py-3 font-medium ${
                    isSmc ? 'text-emerald-400' : 'text-slate-300'
                  }`}
                >
                  {ENGINE_LABELS[s.engine_name] || s.engine_name}
                  {isSmc && ' ★'}
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {(s.detection_rate * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {f1Data ? (f1Data.precision * 100).toFixed(1) : '-'}%
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {f1Data ? (f1Data.recall * 100).toFixed(1) : '-'}%
                </td>
                <td className={`px-4 py-3 font-medium ${isSmc ? 'text-emerald-300' : 'text-slate-400'}`}>
                  {f1Data ? f1Data.f1.toFixed(3) : '-'}
                </td>
                <td className="px-4 py-3 text-slate-400">{calib.toFixed(3)}</td>
                <td className="px-4 py-3 text-slate-400">{s.mean_latency_ms.toFixed(2)}ms</td>
                <td className="px-4 py-3 text-slate-400">{s.positive_segments}</td>
                <td className="px-4 py-3 text-slate-400">
                  {(s.false_positive_estimate * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {(s.noise_rejection_rate * 100).toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
