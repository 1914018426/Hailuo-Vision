"""
DataLab REST API 与 WebSocket 路由

提供录制控制、消融实验管理和结果导出功能。
"""

import asyncio
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.datalab.models import (
    RecordingSession,
    AblationExperiment,
    AnalysisReport,
    ManualLabel,
    ExperimentType,
)
from app.datalab.persistence import DataLabStorage
from app.datalab.recorder import GestureRecorder
from app.datalab.ablation import AblationRunner
from app.datalab.analyzer import AblationAnalyzer
from app.datalab.video_importer import VideoImporter
from app.datalab.charts import (
    svg_bar_chart,
    svg_line_chart,
    svg_heatmap,
    svg_grouped_bar_chart,
    svg_radar_chart,
    svg_xy_lines_chart,
    svg_to_png,
    ENGINE_COLORS,
    ENGINE_LABELS_SHORT,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(prefix="/api/datalab")

# ------------------------------------------------------------------
# Global state (injected from main.py)
# ------------------------------------------------------------------

_storage: Optional[DataLabStorage] = None
_recorder: Optional[GestureRecorder] = None
_runner: Optional[AblationRunner] = None
_analyzer: Optional[AblationAnalyzer] = None
_importer: Optional[VideoImporter] = None

# WebSocket 连接池
_ws_connections: List[WebSocket] = []


def set_storage(storage: DataLabStorage) -> None:
    global _storage
    _storage = storage


def set_recorder(recorder: GestureRecorder) -> None:
    global _recorder
    _recorder = recorder


def _get_storage() -> DataLabStorage:
    if _storage is None:
        raise HTTPException(status_code=503, detail="DataLab 存储未初始化")
    return _storage


def _get_recorder() -> GestureRecorder:
    if _recorder is None:
        raise HTTPException(status_code=503, detail="DataLab 录制器未初始化")
    return _recorder


def _get_runner() -> AblationRunner:
    global _runner
    if _runner is None:
        if _storage is None:
            raise HTTPException(status_code=503, detail="DataLab 存储未初始化")
        _runner = AblationRunner(_storage)
    return _runner


def _get_analyzer() -> AblationAnalyzer:
    global _analyzer
    if _analyzer is None:
        if _storage is None:
            raise HTTPException(status_code=503, detail="DataLab 存储未初始化")
        _analyzer = AblationAnalyzer(_storage)
    return _analyzer


def _get_importer() -> VideoImporter:
    global _importer
    if _importer is None:
        if _storage is None:
            raise HTTPException(status_code=503, detail="DataLab 存储未初始化")
        _importer = VideoImporter(_storage)
    return _importer


# ------------------------------------------------------------------
# Pydantic Request/Response Models
# ------------------------------------------------------------------

class StartRecordingRequest(BaseModel):
    camera_id: str = Field(..., description="摄像头标识")
    trigger_mode: str = Field(default="manual", description="manual / auto_gesture / auto_continuous")
    save_video: bool = Field(default=True, description="是否保存视频文件")


class LabelRecordingRequest(BaseModel):
    label: str = Field(..., description="positive / negative / unlabeled")
    notes: str = Field(default="", description="备注")


class StartExperimentRequest(BaseModel):
    recording_id: str = Field(..., description="录制会话 ID")
    experiment_type: str = Field(
        default="engine_comparison",
        description="engine_comparison / component_ablation / threshold_sweep / scenario_analysis",
    )
    engine_names: Optional[List[str]] = Field(default=None, description="引擎列表，默认全部")
    threshold_range: Optional[List[float]] = Field(
        default=None, description="阈值扫描范围，如 [0.3, 0.5, 0.7, 0.9]"
    )


class StartFullSuiteRequest(BaseModel):
    positive_recording_ids: List[str] = Field(..., description="正样本录制 ID 列表（至少一个，包含 waving 动作）")
    negative_recording_ids: List[str] = Field(..., description="负样本录制 ID 列表（至少一个，不包含 waving 动作）")


class ImportVideoRequest(BaseModel):
    video_path: str = Field(..., description="视频文件路径（容器内绝对路径，如 /app/datasets/hmdb51_extracted/wave/xxx.avi）")
    camera_id: str = Field(default="imported", description="来源标识")
    label: str = Field(default="unlabeled", description="positive / negative / unlabeled")
    notes: str = Field(default="", description="备注")


class MessageResponse(BaseModel):
    message: str


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@router.get("/status", tags=["DataLab"])
async def get_status() -> Dict[str, Any]:
    """获取 DataLab 当前状态。"""
    recorder = _get_recorder()
    runner = _get_runner()
    status = {
        "recording": recorder.get_status().__dict__,
        "experiment": None,
    }
    progress = runner.get_progress()
    if progress:
        status["experiment"] = progress.model_dump()
    return status


# ------------------------------------------------------------------
# Recordings
# ------------------------------------------------------------------

@router.post("/recordings/start", response_model=RecordingSession, tags=["DataLab"])
async def start_recording(req: StartRecordingRequest) -> RecordingSession:
    """开始录制。"""
    recorder = _get_recorder()
    return await recorder.start(
        camera_id=req.camera_id,
        trigger_mode=req.trigger_mode,
        save_video=req.save_video,
    )


@router.post("/recordings/{session_id}/stop", response_model=RecordingSession, tags=["DataLab"])
async def stop_recording(session_id: str) -> RecordingSession:
    """停止录制。"""
    recorder = _get_recorder()
    # 多摄像头：根据 session_id 找到对应的 camera_id 再停止
    target_camera_id: Optional[str] = None
    for cam_id, sess in recorder._sessions.items():
        if sess.id == session_id:
            target_camera_id = cam_id
            break
    if target_camera_id is None:
        raise HTTPException(status_code=400, detail="没有进行中的录制或 ID 不匹配")
    session = await recorder.stop(target_camera_id)
    if session is None or session.id != session_id:
        raise HTTPException(status_code=400, detail="停止录制失败")
    return session


@router.get("/recordings", response_model=List[RecordingSession], tags=["DataLab"])
async def list_recordings() -> List[RecordingSession]:
    """列出所有录制会话。"""
    storage = _get_storage()
    return storage.list_recordings()


@router.get("/recordings/{session_id}", response_model=RecordingSession, tags=["DataLab"])
async def get_recording(session_id: str) -> RecordingSession:
    """获取单个录制会话。"""
    storage = _get_storage()
    session = storage.get_recording(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="录制不存在")
    return session


@router.post("/recordings/{session_id}/label", response_model=RecordingSession, tags=["DataLab"])
async def label_recording(session_id: str, req: LabelRecordingRequest) -> RecordingSession:
    """为录制打标签。"""
    storage = _get_storage()
    session = storage.get_recording(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="录制不存在")
    session.manual_label = req.label
    session.notes = req.notes
    storage.update_recording(session)
    return session


@router.delete("/recordings/{session_id}", response_model=MessageResponse, tags=["DataLab"])
async def delete_recording(session_id: str) -> MessageResponse:
    """删除录制。"""
    storage = _get_storage()
    ok = storage.delete_recording(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="录制不存在")
    return MessageResponse(message=f"录制 {session_id} 已删除")


@router.post("/recordings/import-video", response_model=RecordingSession, tags=["DataLab"])
async def import_video(req: ImportVideoRequest) -> RecordingSession:
    """从本地视频文件导入为 DataLab 录制素材（供离线消融实验使用）。"""
    importer = _get_importer()
    # 使用线程池执行 CPU 密集型离线处理，避免阻塞事件循环
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(
        None,
        importer.import_video,
        req.video_path,
        req.camera_id,
        req.label,
        req.notes,
    )
    return session


@router.post("/recordings/upload-video", response_model=RecordingSession, tags=["DataLab"])
async def upload_video(
    file: UploadFile = File(..., description="上传的视频文件"),
    label: str = Form(default="unlabeled", description="positive / negative / unlabeled"),
    notes: str = Form(default="", description="备注"),
) -> RecordingSession:
    """从客户端上传视频文件并导入为 DataLab 录制素材。"""
    import tempfile
    import shutil

    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        importer = _get_importer()
        loop = asyncio.get_running_loop()
        session = await loop.run_in_executor(
            None,
            importer.import_video,
            tmp_path,
            "uploaded",
            label,
            notes,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return session


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------

@router.post("/experiments/start", response_model=AblationExperiment, tags=["DataLab"])
async def start_experiment(req: StartExperimentRequest) -> AblationExperiment:
    """开始消融实验。"""
    runner = _get_runner()
    return await runner.run_experiment(
        recording_id=req.recording_id,
        experiment_type=req.experiment_type,
        engine_names=req.engine_names,
        threshold_range=req.threshold_range,
    )


@router.post("/experiments/start-full-suite", response_model=AblationExperiment, tags=["DataLab"])
async def start_full_suite(req: StartFullSuiteRequest) -> AblationExperiment:
    """一键启动全套消融实验（引擎对比、组件消融、阈值扫描、场景分析），返回父实验。

    必须同时提供至少一个正样本（包含 waving）和至少一个负样本（不包含 waving），
    以便分别评估召回率与误检率，得出科学结论。
    """
    runner = _get_runner()
    return await runner.run_full_suite(
        req.positive_recording_ids,
        req.negative_recording_ids,
    )


@router.post("/experiments/{exp_id}/stop", response_model=AblationExperiment, tags=["DataLab"])
async def stop_experiment(exp_id: str) -> AblationExperiment:
    """停止/取消实验。"""
    runner = _get_runner()
    exp = runner.get_progress()
    if exp is None or exp.id != exp_id:
        raise HTTPException(status_code=400, detail="没有运行中的实验或 ID 不匹配")
    await runner.cancel()
    # 等待一小段时间让后台任务检测到取消
    await asyncio.sleep(0.5)
    return exp


@router.get("/experiments", response_model=List[AblationExperiment], tags=["DataLab"])
async def list_experiments() -> List[AblationExperiment]:
    """列出所有实验。"""
    storage = _get_storage()
    return storage.list_experiments()


@router.get("/experiments/{exp_id}", response_model=AblationExperiment, tags=["DataLab"])
async def get_experiment(exp_id: str) -> AblationExperiment:
    """获取单个实验。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    return exp


@router.delete("/experiments/{exp_id}", response_model=MessageResponse, tags=["DataLab"])
async def delete_experiment(exp_id: str) -> MessageResponse:
    """删除实验（级联删除子实验）。"""
    storage = _get_storage()
    ok = storage.delete_experiment(exp_id)
    if not ok:
        raise HTTPException(status_code=404, detail="实验不存在")
    return MessageResponse(message=f"实验 {exp_id} 已删除")


@router.get("/experiments/{exp_id}/report", response_model=AnalysisReport, tags=["DataLab"])
async def get_report(exp_id: str) -> AnalysisReport:
    """获取实验分析报告（若不存在则实时生成）。"""
    storage = _get_storage()
    report = storage.get_report(exp_id)
    if report is None:
        analyzer = _get_analyzer()
        exp = storage.get_experiment(exp_id)
        if exp is not None and exp.experiment_type.value == "full_suite":
            report = analyzer.analyze_full_suite(exp_id)
        else:
            report = analyzer.analyze_experiment(exp_id)
    return report


@router.get("/experiments/{exp_id}/export/csv", tags=["DataLab"])
async def export_csv(exp_id: str) -> Dict[str, str]:
    """导出实验 CSV。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    import tempfile
    path = tempfile.mktemp(suffix=".csv")
    storage.export_experiment_csv(exp_id, path)
    return {"download_path": path, "message": "CSV 已生成"}


@router.get("/experiments/{exp_id}/export/json", tags=["DataLab"])
async def export_json(exp_id: str) -> Dict[str, str]:
    """导出实验 JSON。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    storage.export_experiment_json(exp_id, path)
    return {"download_path": path, "message": "JSON 已生成"}


@router.get("/experiments/{exp_id}/export/md", tags=["DataLab"])
async def export_markdown(exp_id: str):
    """导出实验分析报告为 Markdown 文件（含丰富表格与文本可视化）。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    report = storage.get_report(exp_id)
    if report is None:
        analyzer = _get_analyzer()
        if exp.experiment_type == ExperimentType.FULL_SUITE:
            report = analyzer.analyze_full_suite(exp_id)
        else:
            report = analyzer.analyze_experiment(exp_id)
    md = _build_enhanced_markdown(report)
    from fastapi.responses import PlainTextResponse
    filename = f"experiment_{exp_id}_report.md"
    return PlainTextResponse(
        md,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        media_type="text/markdown; charset=utf-8",
    )


def _build_enhanced_markdown(report: AnalysisReport) -> str:
    """生成包含丰富 SVG 图表、数据表格的增强版 Markdown 报告。"""
    from datetime import datetime

    lines: List[str] = []
    lines.append(f"# 实验分析报告 — {report.experiment_id}\n")
    lines.append(f"- **录制素材**: `{report.recording_id}`")
    lines.append(f"- **实验类型**: {report.experiment_type.value if report.experiment_type else 'engine_comparison'}")
    lines.append(f"- **总帧数**: {report.total_frames}")
    lines.append(f"- **共识基线 waving 帧**: {report.consensus_baseline_frames}")
    lines.append(
        f"- **生成时间**: {datetime.fromtimestamp(report.generated_at).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    # 原始结论
    lines.append(report.conclusion_markdown)
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------
    # 引擎统计 — 柱状图 + 表格
    # ------------------------------------------------------------------
    if report.engine_stats:
        lines.append("## 引擎统计详表\n")

        # 检测率柱状图
        detection_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                "value": s.detection_rate * 100,
                "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
            }
            for s in report.engine_stats
        ]
        lines.append("### 检测率对比\n")
        lines.append('<div align="center">')
        lines.append(svg_bar_chart(detection_data, "检测率对比", value_fmt="{:.1f}%"))
        lines.append("</div>\n")

        # F1 柱状图
        f1_data = []
        for s in report.engine_stats:
            prf = report.precision_recall_f1.get(s.engine_name, {})
            f1_data.append(
                {
                    "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                    "value": prf.get("f1", 0) * 100,
                    "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
                }
            )
        if any(d["value"] > 0 for d in f1_data):
            lines.append("### F1 得分对比\n")
            lines.append('<div align="center">')
            lines.append(svg_bar_chart(f1_data, "F1 得分对比", value_fmt="{:.1f}"))
            lines.append("</div>\n")

        # 置信度校准度柱状图（替代平均置信度，避免与精度混淆）
        calib_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                "value": report.calibration_scores.get(s.engine_name, 0.0) * 100,
                "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
            }
            for s in report.engine_stats
        ]
        lines.append("### 置信度校准度对比\n")
        lines.append('<div align="center">')
        lines.append(svg_bar_chart(calib_data, "置信度校准度对比", value_fmt="{:.1f}"))
        lines.append("</div>\n")

        # 雷达图（效率维度使用 inference_count 代替 mean_latency_ms，避免异步推理的虚假延迟）
        max_detection = max(s.detection_rate for s in report.engine_stats) or 0.001
        max_inference = max(s.inference_count for s in report.engine_stats) or 0.001
        max_fp = max(s.false_positive_estimate for s in report.engine_stats) or 0.001
        # 校准度已天然在 0~1，无需额外归一化
        max_calibration = max(report.calibration_scores.values()) or 0.001
        dimensions = ["检测率", "置信度校准度", "鲁棒性", "资源效率", "精度"]
        radar_data = []
        for i, dim_key in enumerate(["detection_rate", "calibration", "robustness", "efficiency", "precision"]):
            row: Dict[str, Any] = {}
            for s in report.engine_stats:
                if dim_key == "detection_rate":
                    row[s.engine_name] = min(1.0, s.detection_rate / max_detection)
                elif dim_key == "calibration":
                    row[s.engine_name] = min(1.0, report.calibration_scores.get(s.engine_name, 0.0) / max_calibration)
                elif dim_key == "robustness":
                    row[s.engine_name] = max(0.0, 1.0 - s.noise_rejection_rate)
                elif dim_key == "efficiency":
                    # inference_count 越少 = 资源效率越高（MiniCPM 推理次数是真实 GPU 成本）
                    row[s.engine_name] = max(0.0, 1.0 - s.inference_count / max_inference)
                elif dim_key == "precision":
                    row[s.engine_name] = max(0.0, 1.0 - s.false_positive_estimate / max_fp)
            radar_data.append(row)
        radar_colors = {s.engine_name: ENGINE_COLORS.get(s.engine_name, "#38bdf8") for s in report.engine_stats}
        radar_labels = {s.engine_name: ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name) for s in report.engine_stats}
        lines.append("### 多维度能力雷达图\n")
        lines.append('<div align="center">')
        lines.append(svg_radar_chart(
            radar_data,
            dimensions,
            list(radar_colors.keys()),
            series_colors=radar_colors,
            series_labels=radar_labels,
            title="多维度能力雷达图",
        ))
        lines.append("</div>\n")

        # 数据表格
        lines.append(
            "| 引擎 | 检测率 | Precision | Recall | F1 | 校准度 | 平均耗时 | 片段数 | 估算误检率 | 静止拒绝率 |"
        )
        lines.append(
            "|------|--------|-----------|--------|----|--------|---------|--------|------------|------------|"
        )
        for s in report.engine_stats:
            prf = report.precision_recall_f1.get(s.engine_name, {})
            calib = report.calibration_scores.get(s.engine_name, 0.0)
            lines.append(
                f"| {s.engine_name} | {(s.detection_rate * 100):.1f}% | "
                f"{prf.get('precision', 0):.3f} | {prf.get('recall', 0):.3f} | {prf.get('f1', 0):.3f} | "
                f"{calib:.3f} | {s.mean_latency_ms:.2f}ms | {s.positive_segments} | "
                f"{(s.false_positive_estimate * 100):.1f}% | {(s.noise_rejection_rate * 100):.1f}% |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # PR / ROC 曲线 — 正确绘制：PR 为 Precision-Recall，ROC 为 TPR-FPR
    # ------------------------------------------------------------------
    # 辅助：若旧数据 engine_name 为空，尝试从 report 推断
    def _infer_eng_name_pdf(fallback: str = "default") -> str:
        if report.engine_stats and len(report.engine_stats) == 1:
            return report.engine_stats[0].engine_name
        if report.precision_recall_f1 and len(report.precision_recall_f1) == 1:
            return list(report.precision_recall_f1.keys())[0]
        return fallback

    if report.pr_curve:
        lines.append("## PR 曲线\n")
        pr_series: Dict[str, List[tuple]] = {}
        for p in report.pr_curve:
            eng = getattr(p, "engine_name", "") or _infer_eng_name_pdf("default")
            if eng not in pr_series:
                pr_series[eng] = []
            pr_series[eng].append((p.recall, p.precision))
        lines.append('<div align="center">')
        lines.append(
            svg_xy_lines_chart(
                pr_series,
                title="PR 曲线",
                x_label="Recall",
                y_label="Precision",
                series_colors=ENGINE_COLORS,
                series_labels=ENGINE_LABELS_SHORT,
                width=600,
                height=520,
            )
        )
        lines.append("</div>\n")

        lines.append("### PR 曲线数据\n")
        lines.append("| 引擎 | 阈值 | Precision | Recall | F1 | TPR | FPR |")
        lines.append("|------|------|-----------|--------|----|-----|-----|")
        for p in report.pr_curve:
            lines.append(
                f"| {getattr(p, 'engine_name', '') or _infer_eng_name_pdf('default')} | {p.threshold:.2f} | {p.precision:.4f} | "
                f"{p.recall:.4f} | {p.f1:.4f} | {p.tpr:.4f} | {p.fpr:.4f} |"
            )
        lines.append("")

    if report.roc_curve:
        lines.append("## ROC 曲线\n")
        roc_series: Dict[str, List[tuple]] = {}
        for p in report.roc_curve:
            eng = getattr(p, "engine_name", "") or _infer_eng_name_pdf("default")
            if eng not in roc_series:
                roc_series[eng] = []
            roc_series[eng].append((p.fpr, p.tpr))
        lines.append('<div align="center">')
        lines.append(
            svg_xy_lines_chart(
                roc_series,
                title="ROC 曲线",
                x_label="FPR",
                y_label="TPR",
                series_colors=ENGINE_COLORS,
                series_labels=ENGINE_LABELS_SHORT,
                width=600,
                height=520,
                show_diagonal=True,
            )
        )
        lines.append("</div>\n")

        lines.append("### ROC 曲线数据\n")
        lines.append("| 引擎 | 阈值 | TPR | FPR |")
        lines.append("|------|------|-----|-----|")
        for p in report.roc_curve:
            lines.append(
                f"| {getattr(p, 'engine_name', '')} | {p.threshold:.2f} | {p.tpr:.4f} | {p.fpr:.4f} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # 组件消融 — 柱状图 + 表格 + 说明
    # ------------------------------------------------------------------
    if report.component_contributions:
        lines.append("## 组件消融分析\n")
        comp_data = [
            {
                "label": c.component_name[:12],
                "value": c.contribution_score,
                "color": "#34d399" if c.contribution_score > 0 else "#f87171",
            }
            for c in report.component_contributions
        ]
        lines.append('<div align="center">')
        lines.append(svg_bar_chart(comp_data, "组件贡献分（正 = 提升性能）", value_fmt="{:.1f}%"))
        lines.append("</div>\n")

        lines.append("### 组件消融详表\n")
        lines.append("| 组件 | 完整 F1 | 消融后 F1 | 贡献分 |")
        lines.append("|------|---------|----------|--------|")
        for c in report.component_contributions:
            sign = "+" if c.contribution_score > 0 else ""
            lines.append(
                f"| {c.component_name} | {c.full_f1:.3f} | {c.ablated_f1:.3f} | {sign}{c.contribution_score:.1f}% |"
            )
        lines.append("")

        lines.append("### 组件功能说明\n")
        for c in report.component_contributions:
            lines.append(f"**{c.component_name}**")
            if c.component_description:
                lines.append(f"> {c.component_description}")
            lines.append("")

    # ------------------------------------------------------------------
    # 场景分析 — 分组柱状图 + 表格
    # ------------------------------------------------------------------
    if report.scenario_stats:
        lines.append("## 场景分析\n")
        scenario_types = sorted({s.scenario_type for s in report.scenario_stats})
        for stype in scenario_types:
            scenarios = [s for s in report.scenario_stats if s.scenario_type == stype]
            if not scenarios:
                continue
            engines = list(scenarios[0].engine_results.keys())[:5]
            data = []
            for s in scenarios:
                row: Dict[str, Any] = {"name": s.scenario_name}
                for e in engines:
                    row[e] = s.engine_results.get(e, {}).get("f1", 0) * 100
                data.append(row)
            colors = {e: ENGINE_COLORS.get(e, "#38bdf8") for e in engines}
            labels = {e: ENGINE_LABELS_SHORT.get(e, e) for e in engines}
            type_label = {"velocity": "速度", "distance": "距离", "hand": "左右手"}.get(stype, stype)
            lines.append(f"### {type_label}场景 F1 对比\n")
            lines.append('<div align="center">')
            lines.append(
                svg_grouped_bar_chart(
                    data,
                    group_key="name",
                    series_keys=engines,
                    series_colors=colors,
                    series_labels=labels,
                    title=f"场景分析 — {type_label}",
                )
            )
            lines.append("</div>\n")

        lines.append("### 场景分析详表\n")
        lines.append("| 场景 | 类型 | 总帧 | GT waving | 引擎 | Precision | Recall | F1 |")
        lines.append("|------|------|------|-----------|------|-----------|--------|----|")
        for ss in report.scenario_stats:
            for eng, vals in ss.engine_results.items():
                lines.append(
                    f"| {ss.scenario_name} | {ss.scenario_type} | {ss.total_frames} | {ss.ground_truth_waving} | "
                    f"{eng} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} |"
                )
        lines.append("")

    # ------------------------------------------------------------------
    # 时序一致性指标
    # ------------------------------------------------------------------
    if report.temporal_metrics:
        lines.append("## 时序一致性指标\n")
        lines.append("| 引擎 | 响应延迟(帧) | 碎片化率(次/s) | 平均片段长度(帧) | 稳定性 CV |")
        lines.append("|------|-------------|---------------|-----------------|----------|")
        for m in report.temporal_metrics:
            lines.append(
                f"| {m.engine_name} | {m.response_latency_mean:.1f}±{m.response_latency_std:.1f} | "
                f"{m.fragmentation_rate:.1f} | {m.avg_positive_duration:.1f} | {m.detection_stability_cv:.4f} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # 一致率矩阵 — 热力图 + 表格
    # ------------------------------------------------------------------
    if report.agreement_matrix.engine_names:
        lines.append("## 引擎一致率矩阵\n")
        lines.append('<div align="center">')
        lines.append(
            svg_heatmap(
                report.agreement_matrix.matrix,
                report.agreement_matrix.engine_names,
                "引擎一致率矩阵",
            )
        )
        lines.append("</div>\n")

        names = report.agreement_matrix.engine_names
        header = "| | " + " | ".join(names) + " |"
        lines.append(header)
        sep = "|---|" + "---|" * len(names)
        lines.append(sep)
        for a in names:
            row = f"| {a} |"
            for b in names:
                val = report.agreement_matrix.matrix.get(a, {}).get(b, 0)
                row += f" {(val * 100):.1f}% |"
            lines.append(row)
        lines.append("")

    # ------------------------------------------------------------------
    # STH 优势指标
    # ------------------------------------------------------------------
    adv = report.simple_minicpm_advantage
    if adv.overall_score > 0:
        lines.append("## Simple+MiniCPM 优势量化\n")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| vs Simple 精度提升 | +{adv.vs_simple_precision_gain:.1f}% |")
        lines.append(f"| vs MiniCPM 召回提升 | +{adv.vs_minicpm_recall_gain:.1f}% |")
        lines.append(f"| 噪声拒绝得分 | {adv.noise_rejection_score:.1f}% |")
        lines.append(f"| 推理效率增益 | {adv.latency_efficiency_gain:.1f}% |")
        lines.append(f"| **综合得分** | **{adv.overall_score:.1f}** |")
        lines.append("")

    lines.append("---")
    lines.append("*报告由 DataLab 消融实验系统自动生成*")
    return "\n".join(lines)


def _generate_all_chart_svgs(report: AnalysisReport) -> Dict[str, str]:
    """从报告生成所有图表 SVG，返回 {文件名: SVG字符串}。"""
    charts: Dict[str, str] = {}

    # 1. 检测率柱状图
    if report.engine_stats:
        detection_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                "value": s.detection_rate * 100,
                "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
            }
            for s in report.engine_stats
        ]
        charts["01_detection_rate.svg"] = svg_bar_chart(detection_data, "检测率对比", value_fmt="{:.1f}%")

        # 2. F1 柱状图
        f1_data = []
        for s in report.engine_stats:
            prf = report.precision_recall_f1.get(s.engine_name, {})
            f1_data.append(
                {
                    "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                    "value": prf.get("f1", 0) * 100,
                    "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
                }
            )
        if any(d["value"] > 0 for d in f1_data):
            charts["02_f1_score.svg"] = svg_bar_chart(f1_data, "F1 得分对比", value_fmt="{:.1f}")

        # 3. 置信度校准度柱状图（替代平均置信度，避免与精度混淆）
        calib_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name),
                "value": report.calibration_scores.get(s.engine_name, 0.0) * 100,
                "color": ENGINE_COLORS.get(s.engine_name, "#38bdf8"),
            }
            for s in report.engine_stats
        ]
        charts["03_calibration.svg"] = svg_bar_chart(calib_data, "置信度校准度对比", value_fmt="{:.1f}")

        # 3b. 雷达图（效率维度使用 inference_count 代替 mean_latency_ms）
        max_detection = max(s.detection_rate for s in report.engine_stats) or 0.001
        max_inference = max(s.inference_count for s in report.engine_stats) or 0.001
        # 精度维度使用 cross-sample precision（与前端雷达图保持一致）
        prf_precision = {
            s.engine_name: report.precision_recall_f1.get(s.engine_name, {}).get("precision", 0.0)
            for s in report.engine_stats
        }
        max_precision = max(prf_precision.values()) or 0.001
        # 校准度已天然在 0~1，无需额外归一化
        max_calibration = max(report.calibration_scores.values()) or 0.001
        dimensions = ["检测率", "置信度校准度", "鲁棒性", "资源效率", "精度"]
        radar_data = []
        for i, dim_key in enumerate(["detection_rate", "calibration", "robustness", "efficiency", "precision"]):
            row: Dict[str, Any] = {}
            for s in report.engine_stats:
                if dim_key == "detection_rate":
                    row[s.engine_name] = min(1.0, s.detection_rate / max_detection)
                elif dim_key == "calibration":
                    row[s.engine_name] = min(1.0, report.calibration_scores.get(s.engine_name, 0.0) / max_calibration)
                elif dim_key == "robustness":
                    row[s.engine_name] = max(0.0, 1.0 - s.noise_rejection_rate)
                elif dim_key == "efficiency":
                    # inference_count 越少 = 资源效率越高
                    row[s.engine_name] = max(0.0, 1.0 - s.inference_count / max_inference)
                elif dim_key == "precision":
                    row[s.engine_name] = min(1.0, prf_precision.get(s.engine_name, 0.0) / max_precision)
            radar_data.append(row)
        radar_colors = {s.engine_name: ENGINE_COLORS.get(s.engine_name, "#38bdf8") for s in report.engine_stats}
        radar_labels = {s.engine_name: ENGINE_LABELS_SHORT.get(s.engine_name, s.engine_name) for s in report.engine_stats}
        charts["03b_radar_comparison.svg"] = svg_radar_chart(
            radar_data,
            dimensions,
            list(radar_colors.keys()),
            series_colors=radar_colors,
            series_labels=radar_labels,
            title="多维度能力雷达图",
        )

    # 辅助：若旧数据 engine_name 为空，尝试从 report 推断
    def _infer_eng_name(fallback: str = "default") -> str:
        if report.engine_stats and len(report.engine_stats) == 1:
            return report.engine_stats[0].engine_name
        if report.precision_recall_f1 and len(report.precision_recall_f1) == 1:
            return list(report.precision_recall_f1.keys())[0]
        return fallback

    # 4. PR 曲线 — 横轴 Recall，纵轴 Precision，每引擎一条线
    if report.pr_curve:
        pr_series: Dict[str, List[tuple]] = {}
        for p in report.pr_curve:
            eng = getattr(p, "engine_name", "") or _infer_eng_name("default")
            if eng not in pr_series:
                pr_series[eng] = []
            pr_series[eng].append((p.recall, p.precision))
        charts["04_pr_curve.svg"] = svg_xy_lines_chart(
            pr_series,
            title="PR 曲线",
            x_label="Recall",
            y_label="Precision",
            series_colors=ENGINE_COLORS,
            series_labels=ENGINE_LABELS_SHORT,
            width=600,
            height=520,
        )

    # 5. ROC 曲线 — 横轴 FPR，纵轴 TPR，每引擎一条线，带随机猜测对角线
    if report.roc_curve:
        roc_series: Dict[str, List[tuple]] = {}
        for p in report.roc_curve:
            eng = getattr(p, "engine_name", "") or _infer_eng_name("default")
            if eng not in roc_series:
                roc_series[eng] = []
            roc_series[eng].append((p.fpr, p.tpr))
        charts["05_roc_curve.svg"] = svg_xy_lines_chart(
            roc_series,
            title="ROC 曲线",
            x_label="FPR",
            y_label="TPR",
            series_colors=ENGINE_COLORS,
            series_labels=ENGINE_LABELS_SHORT,
            width=600,
            height=520,
            show_diagonal=True,
        )

    # 6. 组件消融贡献
    if report.component_contributions:
        comp_data = [
            {
                "label": c.component_name[:12],
                "value": c.contribution_score,
                "color": "#34d399" if c.contribution_score > 0 else "#f87171",
            }
            for c in report.component_contributions
        ]
        charts["06_component_contribution.svg"] = svg_bar_chart(
            comp_data, "组件贡献分（正 = 提升性能）", value_fmt="{:.1f}%"
        )

    # 7. 场景分析分组柱状图
    if report.scenario_stats:
        scenario_types = sorted({s.scenario_type for s in report.scenario_stats})
        for idx, stype in enumerate(scenario_types):
            scenarios = [s for s in report.scenario_stats if s.scenario_type == stype]
            if not scenarios:
                continue
            engines = list(scenarios[0].engine_results.keys())[:5]
            data = []
            for s in scenarios:
                row: Dict[str, Any] = {"name": s.scenario_name}
                for e in engines:
                    row[e] = s.engine_results.get(e, {}).get("f1", 0) * 100
                data.append(row)
            colors = {e: ENGINE_COLORS.get(e, "#38bdf8") for e in engines}
            labels = {e: ENGINE_LABELS_SHORT.get(e, e) for e in engines}
            type_label = {"velocity": "速度", "distance": "距离", "hand": "左右手"}.get(stype, stype)
            fname = f"07_scenario_{stype}_f1.svg"
            # 根据分组数量动态调整宽度，避免柱子过窄
            n_groups = len(data)
            n_series = len(engines)
            chart_width = max(800, min(2400, n_groups * n_series * 70 + 200))
            charts[fname] = svg_grouped_bar_chart(
                data,
                group_key="name",
                series_keys=engines,
                series_colors=colors,
                series_labels=labels,
                title=f"场景分析 — {type_label}",
                width=chart_width,
            )

    # 8. 时序一致性 — 改为三个独立柱状图，避免单点折线图无意义
    if report.temporal_metrics:
        tms = sorted(report.temporal_metrics, key=lambda x: x.engine_name)[:6]
        # 响应延迟柱状图
        latency_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(tm.engine_name, tm.engine_name),
                "value": tm.response_latency_mean,
                "color": ENGINE_COLORS.get(tm.engine_name, "#38bdf8"),
            }
            for tm in tms
        ]
        charts["08a_latency.svg"] = svg_bar_chart(
            latency_data, "响应延迟对比（帧）", value_fmt="{:.1f}"
        )
        # 碎片化率柱状图
        frag_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(tm.engine_name, tm.engine_name),
                "value": tm.fragmentation_rate,
                "color": ENGINE_COLORS.get(tm.engine_name, "#38bdf8"),
            }
            for tm in tms
        ]
        charts["08b_fragmentation.svg"] = svg_bar_chart(
            frag_data, "碎片化率对比（次/s）", value_fmt="{:.1f}"
        )
        # 平均片段长度柱状图
        duration_data = [
            {
                "label": ENGINE_LABELS_SHORT.get(tm.engine_name, tm.engine_name),
                "value": tm.avg_positive_duration,
                "color": ENGINE_COLORS.get(tm.engine_name, "#38bdf8"),
            }
            for tm in tms
        ]
        charts["08c_duration.svg"] = svg_bar_chart(
            duration_data, "平均片段长度对比（帧）", value_fmt="{:.1f}"
        )

    # 9. 一致率矩阵热力图
    if report.agreement_matrix.engine_names:
        charts["09_agreement_heatmap.svg"] = svg_heatmap(
            report.agreement_matrix.matrix,
            report.agreement_matrix.engine_names,
            "引擎一致率矩阵",
        )

    return charts


@router.get("/experiments/{exp_id}/export/charts", tags=["DataLab"])
async def export_charts(exp_id: str):
    """导出实验所有图表为 PNG 打包 ZIP 下载。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    report = storage.get_report(exp_id)
    if report is None:
        analyzer = _get_analyzer()
        if exp.experiment_type == ExperimentType.FULL_SUITE:
            report = analyzer.analyze_full_suite(exp_id)
        else:
            report = analyzer.analyze_experiment(exp_id)

    svgs = _generate_all_chart_svgs(report)
    if not svgs:
        raise HTTPException(status_code=404, detail="该实验没有可导出的图表")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, svg in svgs.items():
            png_name = fname.replace(".svg", ".png")
            png_bytes = svg_to_png(svg, scale=2.0)
            zf.writestr(png_name, png_bytes)

    zip_buffer.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}_charts.zip"'},
    )


# ------------------------------------------------------------------
# PDF 导出
# ------------------------------------------------------------------

def _build_pdf_html(report: AnalysisReport) -> str:
    """为 WeasyPrint 构建 PDF 用的 HTML（白底学术风格，图表内嵌 PNG）。"""
    import base64
    from datetime import datetime

    # 生成所有图表 PNG（base64）
    svgs = _generate_all_chart_svgs(report)
    chart_images: Dict[str, str] = {}
    for fname, svg in svgs.items():
        png_bytes = svg_to_png(svg, scale=2.5)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        key = fname.replace(".svg", "")
        chart_images[key] = f"data:image/png;base64,{b64}"

    # 结论 markdown 转 HTML
    try:
        import markdown as md_lib
        conclusion_html = md_lib.markdown(report.conclusion_markdown, extensions=["tables"])
    except Exception:
        conclusion_html = f"<pre>{report.conclusion_markdown}</pre>"

    # 引擎统计表格
    engine_rows = []
    if report.engine_stats:
        for s in report.engine_stats:
            prf = report.precision_recall_f1.get(s.engine_name, {})
            calib = report.calibration_scores.get(s.engine_name, 0.0)
            engine_rows.append(
                f"<tr>"
                f"<td>{s.engine_name}</td>"
                f"<td>{(s.detection_rate * 100):.1f}%</td>"
                f"<td>{prf.get('precision', 0):.3f}</td>"
                f"<td>{prf.get('recall', 0):.3f}</td>"
                f"<td>{prf.get('f1', 0):.3f}</td>"
                f"<td>{calib:.3f}</td>"
                f"<td>{s.mean_latency_ms:.2f}ms</td>"
                f"<td>{s.positive_segments}</td>"
                f"<td>{(s.false_positive_estimate * 100):.1f}%</td>"
                f"</tr>"
            )

    # 组件消融表格
    comp_rows = []
    for c in report.component_contributions:
        sign = "+" if c.contribution_score > 0 else ""
        comp_rows.append(
            f"<tr>"
            f"<td>{c.component_name}</td>"
            f"<td>{c.full_f1:.3f}</td>"
            f"<td>{c.ablated_f1:.3f}</td>"
            f"<td>{sign}{c.contribution_score:.1f}%</td>"
            f"<td>{c.component_description or ''}</td>"
            f"</tr>"
        )

    # 场景分析表格
    scenario_rows = []
    for ss in report.scenario_stats:
        for eng, vals in ss.engine_results.items():
            scenario_rows.append(
                f"<tr>"
                f"<td>{ss.scenario_name}</td>"
                f"<td>{ss.scenario_type}</td>"
                f"<td>{ss.total_frames}</td>"
                f"<td>{ss.ground_truth_waving}</td>"
                f"<td>{eng}</td>"
                f"<td>{vals['precision']:.3f}</td>"
                f"<td>{vals['recall']:.3f}</td>"
                f"<td>{vals['f1']:.3f}</td>"
                f"</tr>"
            )

    # 时序指标表格
    temporal_rows = []
    for m in report.temporal_metrics:
        temporal_rows.append(
            f"<tr>"
            f"<td>{m.engine_name}</td>"
            f"<td>{m.response_latency_mean:.1f}±{m.response_latency_std:.1f}</td>"
            f"<td>{m.fragmentation_rate:.1f}</td>"
            f"<td>{m.avg_positive_duration:.1f}</td>"
            f"<td>{m.detection_stability_cv:.4f}</td>"
            f"</tr>"
        )

    # 一致率矩阵表格
    agreement_html = ""
    if report.agreement_matrix.engine_names:
        names = report.agreement_matrix.engine_names
        header = "<th></th>" + "".join(f"<th>{n}</th>" for n in names)
        agr_rows = []
        for a in names:
            row = f"<td><strong>{a}</strong></td>"
            for b in names:
                val = report.agreement_matrix.matrix.get(a, {}).get(b, 0)
                row += f"<td>{(val * 100):.1f}%</td>"
            agr_rows.append(f"<tr>{row}</tr>")
        agreement_html = f"""
        <h2>引擎一致率矩阵</h2>
        <img src="{chart_images.get('09_agreement_heatmap', '')}" class="chart-img" />
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(agr_rows)}</tbody>
        </table>
        """

    # STH 优势
    sth_html = ""
    adv = report.simple_minicpm_advantage
    if adv.overall_score > 0:
        sth_html = f"""
        <h2>Simple+MiniCPM 优势量化</h2>
        <table>
            <thead><tr><th>指标</th><th>数值</th></tr></thead>
            <tbody>
                <tr><td>vs Simple 精度提升</td><td>+{adv.vs_simple_precision_gain:.1f}%</td></tr>
                <tr><td>vs MiniCPM 召回提升</td><td>+{adv.vs_minicpm_recall_gain:.1f}%</td></tr>
                <tr><td>噪声拒绝得分</td><td>{adv.noise_rejection_score:.1f}%</td></tr>
                <tr><td>推理效率增益</td><td>{adv.latency_efficiency_gain:.1f}%</td></tr>
                <tr><td><strong>综合得分</strong></td><td><strong>{adv.overall_score:.1f}</strong></td></tr>
            </tbody>
        </table>
        """

    # 组装图表区块
    chart_sections = []
    if "01_detection_rate" in chart_images:
        chart_sections.append(f'<h2>检测率对比</h2><img src="{chart_images["01_detection_rate"]}" class="chart-img" />')
    if "02_f1_score" in chart_images:
        chart_sections.append(f'<h2>F1 得分对比</h2><img src="{chart_images["02_f1_score"]}" class="chart-img" />')
    if "03_calibration" in chart_images:
        chart_sections.append(f'<h2>置信度校准度对比</h2><img src="{chart_images["03_calibration"]}" class="chart-img" />')
    if "03b_radar_comparison" in chart_images:
        chart_sections.append(f'<h2>多维度能力雷达图</h2><img src="{chart_images["03b_radar_comparison"]}" class="chart-img" />')
    if "04_pr_curve" in chart_images:
        chart_sections.append(f'<h2>PR 曲线</h2><img src="{chart_images["04_pr_curve"]}" class="chart-img" />')
    if "05_roc_curve" in chart_images:
        chart_sections.append(f'<h2>ROC 曲线</h2><img src="{chart_images["05_roc_curve"]}" class="chart-img" />')
    if "06_component_contribution" in chart_images:
        chart_sections.append(f'<h2>组件消融贡献</h2><img src="{chart_images["06_component_contribution"]}" class="chart-img" />')
    for key in sorted(chart_images.keys()):
        if key.startswith("07_scenario_"):
            title = key.replace("07_scenario_", "").replace("_f1", "")
            chart_sections.append(f'<h2>场景分析 — {title}</h2><img src="{chart_images[key]}" class="chart-img" />')
    # 时序一致性三个独立柱状图
    for key in ["08a_latency", "08b_fragmentation", "08c_duration"]:
        if key in chart_images:
            title = {"08a_latency": "响应延迟对比", "08b_fragmentation": "碎片化率对比", "08c_duration": "平均片段长度对比"}.get(key, key)
            chart_sections.append(f'<h2>{title}</h2><img src="{chart_images[key]}" class="chart-img" />')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>实验分析报告 — {report.experiment_id}</title>
<style>
@page {{
    size: A4;
    margin: 2cm;
    @bottom-center {{ content: counter(page); font-size: 9pt; color: #666; }}
}}
body {{
    font-family: "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei", sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fff;
}}
h1 {{ font-size: 18pt; color: #0f172a; border-bottom: 2px solid #38bdf8; padding-bottom: 8px; margin-top: 0; }}
h2 {{ font-size: 14pt; color: #1e293b; margin-top: 24px; margin-bottom: 12px; border-left: 4px solid #38bdf8; padding-left: 10px; }}
h3 {{ font-size: 12pt; color: #334155; margin-top: 18px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
}}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
th {{ background: #f1f5f9; font-weight: 600; }}
tr:nth-child(even) {{ background: #f8fafc; }}
.chart-img {{
    display: block;
    max-width: 100%;
    margin: 16px auto;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.meta {{
    background: #f8fafc;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 10pt;
    color: #475569;
}}
.meta p {{ margin: 4px 0; }}
pre {{ background: #f8fafc; padding: 12px; border-radius: 6px; overflow-x: auto; }}
blockquote {{
    border-left: 4px solid #94a3b8;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8fafc;
    color: #475569;
}}
ul, ol {{ margin: 8px 0; padding-left: 24px; }}
li {{ margin: 4px 0; }}
</style>
</head>
<body>
<h1>实验分析报告 — {report.experiment_id}</h1>
<div class="meta">
    <p><strong>录制素材:</strong> {report.recording_id}</p>
    <p><strong>实验类型:</strong> {report.experiment_type.value if report.experiment_type else 'engine_comparison'}</p>
    <p><strong>总帧数:</strong> {report.total_frames}</p>
    <p><strong>共识基线 waving 帧:</strong> {report.consensus_baseline_frames}</p>
    <p><strong>生成时间:</strong> {datetime.fromtimestamp(report.generated_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

{conclusion_html}

<h2>引擎统计详表</h2>
<table>
<thead><tr>
    <th>引擎</th><th>检测率</th><th>Precision</th><th>Recall</th><th>F1</th>
    <th>校准度</th><th>平均耗时</th><th>片段数</th><th>估算误检率</th>
</tr></thead>
<tbody>{''.join(engine_rows)}</tbody>
</table>

{''.join(chart_sections)}

{agreement_html}

<h2>组件消融详表</h2>
<table>
<thead><tr><th>组件</th><th>完整 F1</th><th>消融后 F1</th><th>贡献分</th><th>说明</th></tr></thead>
<tbody>{''.join(comp_rows)}</tbody>
</table>

<h2>场景分析详表</h2>
<table>
<thead><tr><th>场景</th><th>类型</th><th>总帧</th><th>GT waving</th><th>引擎</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
<tbody>{''.join(scenario_rows)}</tbody>
</table>

<h2>时序一致性指标</h2>
<table>
<thead><tr><th>引擎</th><th>响应延迟(帧)</th><th>碎片化率(次/s)</th><th>平均片段长度(帧)</th><th>稳定性 CV</th></tr></thead>
<tbody>{''.join(temporal_rows)}</tbody>
</table>

{sth_html}

<p style="margin-top:40px; color:#94a3b8; font-size:9pt; text-align:center;">
    报告由 DataLab 消融实验系统自动生成
</p>
</body>
</html>"""
    return html


@router.get("/experiments/{exp_id}/export/pdf", tags=["DataLab"])
async def export_pdf(exp_id: str):
    """导出实验分析报告为 PDF（含图表与数据表格）。"""
    storage = _get_storage()
    exp = storage.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="实验不存在")
    report = storage.get_report(exp_id)
    if report is None:
        analyzer = _get_analyzer()
        if exp.experiment_type == ExperimentType.FULL_SUITE:
            report = analyzer.analyze_full_suite(exp_id)
        else:
            report = analyzer.analyze_experiment(exp_id)

    html = _build_pdf_html(report)
    pdf_buffer = io.BytesIO()
    import weasyprint
    weasyprint.HTML(string=html).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}_report.pdf"'},
    )


# ------------------------------------------------------------------
# WebSocket for real-time progress
# ------------------------------------------------------------------

@router.websocket("/ws/datalab")
async def datalab_websocket(websocket: WebSocket) -> None:
    """WebSocket 实时推送录制/实验进度。"""
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            # 每秒推送一次状态
            recorder = _get_recorder()
            runner = _get_runner()
            status = {
                "recording": recorder.get_status().__dict__,
                "experiment": None,
            }
            progress = runner.get_progress()
            if progress:
                status["experiment"] = progress.model_dump()
            await websocket.send_json(status)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("DataLab WebSocket 异常: %s", e)
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)
