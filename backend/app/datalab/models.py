"""
DataLab Pydantic 数据模型

定义录制会话、消融实验、统计结果等数据结构。
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RecordingTriggerMode(str, Enum):
    """录制触发模式。"""

    MANUAL = "manual"
    AUTO_GESTURE = "auto_gesture"
    AUTO_CONTINUOUS = "auto_continuous"


class ManualLabel(str, Enum):
    """人工标签。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNLABELED = "unlabeled"


class RecordingSession(BaseModel):
    """录制会话元数据。"""

    id: str = Field(..., description="录制会话唯一标识（UUID）")
    camera_id: str = Field(..., description="摄像头标识")
    trigger_mode: RecordingTriggerMode = Field(
        default=RecordingTriggerMode.MANUAL, description="触发模式"
    )
    start_time: float = Field(..., description="开始时间戳（Unix epoch seconds）")
    end_time: Optional[float] = Field(default=None, description="结束时间戳")
    duration_s: Optional[float] = Field(default=None, description="录制时长（秒）")
    frame_count: int = Field(default=0, description="录制的帧数")
    person_count: int = Field(default=0, description="检测到的人物数峰值")
    video_path: Optional[str] = Field(default=None, description="视频文件路径")
    keypoints_path: Optional[str] = Field(default=None, description="关键点序列文件路径")
    tnlf_path: Optional[str] = Field(default=None, description="TNLF 特征文件路径")
    detections_path: Optional[str] = Field(default=None, description="主引擎检测输出文件路径")
    meta_path: str = Field(..., description="元数据文件路径")
    manual_label: ManualLabel = Field(default=ManualLabel.UNLABELED, description="人工标签")
    notes: str = Field(default="", description="备注")
    status: str = Field(default="recording", description="状态: recording / completed / failed")


class FrameSnapshot(BaseModel):
    """单帧快照数据。"""

    frame_idx: int = Field(..., description="帧序号")
    timestamp: float = Field(..., description="帧时间戳")
    camera_id: str = Field(..., description="摄像头标识")
    keypoints: Optional[List[List[float]]] = Field(
        default=None, description="COCO-17 关键点 (x, y, conf)"
    )
    hand_landmarks_left: Optional[List[Tuple[float, float, float]]] = Field(
        default=None, description="左手 MediaPipe 21 点"
    )
    hand_landmarks_right: Optional[List[Tuple[float, float, float]]] = Field(
        default=None, description="右手 MediaPipe 21 点"
    )
    left_palm_normal: Optional[List[float]] = Field(default=None, description="左掌法向量")
    right_palm_normal: Optional[List[float]] = Field(default=None, description="右掌法向量")
    tnlf_features: Optional[Dict[str, Any]] = Field(default=None, description="12 维 TNLF 特征")


class EngineFrameResult(BaseModel):
    """单引擎单帧推理结果。"""

    engine_name: str = Field(..., description="引擎名称")
    gesture: str = Field(default="none", description="识别结果手势")
    confidence: float = Field(default=0.0, description="置信度")
    latency_ms: Optional[float] = Field(default=None, description="推理耗时（毫秒）")


class MultiEngineFrameResult(BaseModel):
    """多引擎单帧对比结果。"""

    frame_idx: int = Field(..., description="帧序号")
    timestamp: float = Field(..., description="时间戳")
    results: Dict[str, EngineFrameResult] = Field(
        default_factory=dict, description="引擎名 -> 结果"
    )
    velocity_left: float = Field(default=0.0, description="左手腕速度")
    velocity_right: float = Field(default=0.0, description="右手腕速度")


class ExperimentType(str, Enum):
    """实验类型。"""

    ENGINE_COMPARISON = "engine_comparison"
    COMPONENT_ABLATION = "component_ablation"
    THRESHOLD_SWEEP = "threshold_sweep"
    SCENARIO_ANALYSIS = "scenario_analysis"
    FULL_SUITE = "full_suite"


class ExperimentStatus(str, Enum):
    """实验状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AblationExperiment(BaseModel):
    """消融实验配置与结果。"""

    id: str = Field(..., description="实验唯一标识（UUID）")
    recording_id: str = Field(..., description="关联的录制会话 ID")
    experiment_type: ExperimentType = Field(
        default=ExperimentType.ENGINE_COMPARISON, description="实验类型"
    )
    engine_names: List[str] = Field(
        default_factory=list, description="参与实验的引擎列表"
    )
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    progress: float = Field(default=0.0, description="进度 0-1")
    current_frame: int = Field(default=0, description="当前处理帧")
    total_frames: int = Field(default=0, description="总帧数")
    error_message: Optional[str] = Field(default=None)
    frame_results_path: Optional[str] = Field(default=None, description="逐帧结果 JSONL 路径")
    engine_stats_path: Optional[str] = Field(default=None, description="引擎统计 JSON 路径")
    report_path: Optional[str] = Field(default=None, description="分析报告路径")
    created_at: float = Field(..., description="创建时间戳")
    completed_at: Optional[float] = Field(default=None)
    # 全量实验父子关系
    parent_id: Optional[str] = Field(default=None, description="父实验 ID（子实验指向父）")
    sub_experiment_ids: List[str] = Field(
        default_factory=list, description="子实验 ID 列表（仅父实验有）"
    )
    # 全量实验正负样本列表（仅父实验有）
    positive_recording_ids: List[str] = Field(
        default_factory=list, description="正样本录制 ID 列表"
    )
    negative_recording_ids: List[str] = Field(
        default_factory=list, description="负样本录制 ID 列表"
    )


class EngineStats(BaseModel):
    """单引擎聚合统计。"""

    engine_name: str = Field(..., description="引擎名称")
    total_frames: int = Field(default=0, description="总帧数")
    waving_frames: int = Field(default=0, description="检测到 waving 的帧数")
    detection_rate: float = Field(default=0.0, description="检测率")
    mean_confidence: float = Field(default=0.0, description="平均置信度（仅 waving 帧）")
    std_confidence: float = Field(default=0.0, description="置信度标准差")
    max_confidence: float = Field(default=0.0, description="最大置信度")
    min_confidence: float = Field(default=0.0, description="最小置信度")
    mean_latency_ms: float = Field(default=0.0, description="平均推理耗时")
    positive_segments: int = Field(default=0, description="连续 waving 片段数")
    false_positive_estimate: float = Field(default=0.0, description="估算误检率（基于共识基线）")
    noise_rejection_rate: float = Field(default=0.0, description="低速度场景拒绝率（velocity < 0.03）")
    inference_count: int = Field(default=0, description="MiniCPM 推理次数（Simple 引擎为 0）")


class AgreementMatrix(BaseModel):
    """引擎间一致率矩阵。"""

    engine_names: List[str] = Field(default_factory=list)
    matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="engine_a -> engine_b -> 一致率"
    )


class SimpleMiniCPMHybridAdvantage(BaseModel):
    """Simple+MiniCPM 混合引擎优势分析。"""

    vs_simple_precision_gain: float = Field(
        default=0.0, description="相比 Simple 的精度增益（百分比）"
    )
    vs_minicpm_recall_gain: float = Field(
        default=0.0, description="相比 MiniCPM 的召回增益（百分比）"
    )
    noise_rejection_score: float = Field(
        default=0.0, description="静止场景误检率（越低越好，此处为相对优势分）"
    )
    latency_efficiency_gain: float = Field(
        default=0.0, description="推理效率增益（百分比）"
    )
    overall_score: float = Field(
        default=0.0, description="综合得分"
    )


class PRCurvePoint(BaseModel):
    """PR/ROC 曲线单点。"""

    threshold: float = Field(..., description="置信度阈值")
    precision: float = Field(default=0.0)
    recall: float = Field(default=0.0)
    f1: float = Field(default=0.0)
    tpr: float = Field(default=0.0, description="真正例率")
    fpr: float = Field(default=0.0, description="假正例率")
    engine_name: str = Field(default="", description="所属引擎名称")


class ComponentContribution(BaseModel):
    """组件消融贡献分析。"""

    component_name: str = Field(..., description="组件名称")
    component_description: str = Field(default="", description="组件中文详细说明")
    full_f1: float = Field(default=0.0, description="完整 STH 的 F1")
    ablated_f1: float = Field(default=0.0, description="消融后的 F1")
    contribution_score: float = Field(
        default=0.0, description="贡献分 (%), 正数表示该组件提升性能"
    )


class ScenarioStats(BaseModel):
    """单场景统计。"""

    scenario_name: str = Field(..., description="场景名称")
    scenario_type: str = Field(..., description="场景类型: velocity/distance/hand/confidence")
    total_frames: int = Field(default=0)
    ground_truth_waving: int = Field(default=0, description="共识基线 waving 帧数")
    engine_results: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="engine_name -> {precision, recall, f1}"
    )


class TemporalMetrics(BaseModel):
    """时序一致性指标。"""

    engine_name: str = Field(..., description="引擎名称")
    response_latency_mean: float = Field(default=0.0, description="平均响应延迟（帧）")
    response_latency_std: float = Field(default=0.0, description="响应延迟标准差")
    fragmentation_rate: float = Field(default=0.0, description="每秒状态翻转次数")
    avg_positive_duration: float = Field(default=0.0, description="平均 waving 片段长度（帧）")
    detection_stability_cv: float = Field(
        default=0.0, description="waving 片段置信度变异系数"
    )


class AnalysisReport(BaseModel):
    """完整分析报告。"""

    experiment_id: str = Field(..., description="实验 ID")
    recording_id: str = Field(..., description="录制会话 ID")
    experiment_type: ExperimentType = Field(
        default=ExperimentType.ENGINE_COMPARISON, description="实验类型"
    )
    total_frames: int = Field(default=0)
    engine_stats: List[EngineStats] = Field(default_factory=list)
    agreement_matrix: AgreementMatrix = Field(default_factory=AgreementMatrix)
    consensus_baseline_frames: int = Field(
        default=0, description="3/5 引擎共识基线 waving 帧数"
    )
    simple_minicpm_advantage: SimpleMiniCPMHybridAdvantage = Field(
        default_factory=SimpleMiniCPMHybridAdvantage
    )
    # 新增：科学指标
    precision_recall_f1: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="engine_name -> {precision, recall, f1}"
    )
    pr_curve: List[PRCurvePoint] = Field(default_factory=list, description="PR 曲线")
    roc_curve: List[PRCurvePoint] = Field(default_factory=list, description="ROC 曲线")
    component_contributions: List[ComponentContribution] = Field(
        default_factory=list, description="组件消融贡献"
    )
    scenario_stats: List[ScenarioStats] = Field(
        default_factory=list, description="场景细分统计"
    )
    temporal_metrics: List[TemporalMetrics] = Field(
        default_factory=list, description="时序一致性"
    )
    calibration_scores: Dict[str, float] = Field(
        default_factory=dict, description="engine_name -> 置信度校准度 [0,1]"
    )
    conclusion_markdown: str = Field(
        default="", description="自动生成的 Markdown 结论"
    )
    generated_at: float = Field(..., description="生成时间戳")
