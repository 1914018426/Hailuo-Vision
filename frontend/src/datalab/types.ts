export type RecordingTriggerMode = 'manual' | 'auto_gesture' | 'auto_continuous';
export type ManualLabel = 'positive' | 'negative' | 'unlabeled';
export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type ExperimentType = 'engine_comparison' | 'component_ablation' | 'threshold_sweep' | 'scenario_analysis' | 'full_suite';

export interface RecordingSession {
  id: string;
  camera_id: string;
  trigger_mode: RecordingTriggerMode;
  start_time: number;
  end_time?: number;
  duration_s?: number;
  frame_count: number;
  person_count: number;
  video_path?: string;
  keypoints_path?: string;
  tnlf_path?: string;
  detections_path?: string;
  meta_path: string;
  manual_label: ManualLabel;
  notes: string;
  status: string;
}

export interface EngineFrameResult {
  engine_name: string;
  gesture: string;
  confidence: number;
  latency_ms?: number;
}

export interface EngineStats {
  engine_name: string;
  total_frames: number;
  waving_frames: number;
  detection_rate: number;
  mean_confidence: number;
  std_confidence: number;
  max_confidence: number;
  min_confidence: number;
  mean_latency_ms: number;
  positive_segments: number;
  false_positive_estimate: number;
  noise_rejection_rate: number;
  inference_count: number;
}

export interface AgreementMatrix {
  engine_names: string[];
  matrix: Record<string, Record<string, number>>;
}

export interface SimpleMiniCPMHybridAdvantage {
  vs_simple_precision_gain: number;
  vs_minicpm_recall_gain: number;
  noise_rejection_score: number;
  latency_efficiency_gain: number;
  overall_score: number;
}

export interface PRCurvePoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  tpr: number;
  fpr: number;
  engine_name: string;
}

export interface ComponentContribution {
  component_name: string;
  component_description?: string;
  full_f1: number;
  ablated_f1: number;
  contribution_score: number;
}

export interface ScenarioStats {
  scenario_name: string;
  scenario_type: string;
  total_frames: number;
  ground_truth_waving: number;
  engine_results: Record<string, { precision: number; recall: number; f1: number }>;
}

export interface TemporalMetrics {
  engine_name: string;
  response_latency_mean: number;
  response_latency_std: number;
  fragmentation_rate: number;
  avg_positive_duration: number;
  detection_stability_cv: number;
}

export interface AblationExperiment {
  id: string;
  recording_id: string;
  experiment_type?: ExperimentType;
  engine_names: string[];
  status: ExperimentStatus;
  progress: number;
  current_frame: number;
  total_frames: number;
  error_message?: string;
  frame_results_path?: string;
  engine_stats_path?: string;
  report_path?: string;
  created_at: number;
  completed_at?: number;
  sub_experiment_ids?: string[];
}

export interface AnalysisReport {
  experiment_id: string;
  recording_id: string;
  experiment_type?: ExperimentType;
  total_frames: number;
  engine_stats: EngineStats[];
  agreement_matrix: AgreementMatrix;
  consensus_baseline_frames: number;
  simple_minicpm_advantage: SimpleMiniCPMHybridAdvantage;
  precision_recall_f1: Record<string, { precision: number; recall: number; f1: number }>;
  pr_curve: PRCurvePoint[];
  roc_curve: PRCurvePoint[];
  component_contributions: ComponentContribution[];
  scenario_stats: ScenarioStats[];
  temporal_metrics: TemporalMetrics[];
  calibration_scores: Record<string, number>;
  conclusion_markdown: string;
  generated_at: number;
}

export interface ActiveSession {
  session_id: string;
  camera_id: string;
  trigger_mode: string;
  frame_count: number;
  duration_s: number;
  person_count_peak: number;
}

export interface RecorderStatus {
  is_recording: boolean;
  session_id?: string;
  camera_id?: string;
  trigger_mode?: string;
  frame_count: number;
  duration_s: number;
  person_count_peak: number;
  segment_count: number;
  sessions: ActiveSession[];
  total_active: number;
}

export interface DataLabStatus {
  recording: RecorderStatus;
  experiment: AblationExperiment | null;
}
