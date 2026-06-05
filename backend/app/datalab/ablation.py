"""
消融实验运行器 — 全面增强版

支持 4 种实验类型：
  1. engine_comparison   — 5 主引擎横向对比
  2. component_ablation  — STH 组件逐一消融
  3. threshold_sweep     — 阈值扫描生成 PR/ROC 曲线
  4. scenario_analysis   — 按速度/距离/左右手分场景统计

组件消融变体通过子类化 + 方法覆盖实现，不修改 gesture.py 原引擎代码。
"""

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

from app.datalab.models import (
    AblationExperiment,
    ExperimentStatus,
    ExperimentType,
    MultiEngineFrameResult,
    EngineFrameResult,
)
from app.datalab.persistence import DataLabStorage
from app.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Simple+MiniCPM 组件消融包装器
# =============================================================================

class _SimpleMiniCPMNoSimpleGate:
    """Simple+MiniCPM 去掉 Simple trigger，直接让 MiniCPM 看到所有帧。"""

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        # 强制 trigger 为 True，绕过 Simple 姿态门
        kwargs = dict(kwargs)
        # 保存原始方法
        original_check = self._base._simple.check_trigger_only
        self._base._simple.check_trigger_only = lambda _kp: True
        try:
            return self._base.recognize(*args, **kwargs)
        finally:
            self._base._simple.check_trigger_only = original_check

    def reset(self):
        self._base.reset()


class _SimpleMiniCPMNoCooldown:
    """Simple+MiniCPM 去掉冷却期，允许更频繁的推理。"""

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        original_cooldown = self._base._cooldown_seconds
        self._base._cooldown_seconds = 0.0
        try:
            return self._base.recognize(*args, **kwargs)
        finally:
            self._base._cooldown_seconds = original_cooldown

    def reset(self):
        self._base.reset()


# =============================================================================
# 主运行器
# =============================================================================

class AblationRunner:
    """消融实验运行器（全面增强版，支持队列）。"""

    def __init__(self, storage: DataLabStorage) -> None:
        self.storage = storage
        self.config = get_config()
        self._current_exp: Optional[AblationExperiment] = None
        self._cancelled: bool = False
        self._lock = asyncio.Lock()
        self._queue: List[Tuple[AblationExperiment, List[str], ExperimentType, Optional[List[float]]]] = []
        self._queue_task: Optional[asyncio.Task] = None
        # 共享 MiniCPM 引擎实例：避免多个实例并发加载模型导致权重损坏/输出乱码
        self._shared_minicpm_engine: Optional[Any] = None

    async def run_experiment(
        self,
        recording_id: str,
        experiment_type: str = "engine_comparison",
        engine_names: Optional[List[str]] = None,
        threshold_range: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
    ) -> AblationExperiment:
        """
        运行消融实验（入队执行）。

        Args:
            recording_id: 录制会话 ID
            experiment_type: engine_comparison / component_ablation / threshold_sweep / scenario_analysis
            engine_names: 要测试的引擎列表（engine_comparison / scenario_analysis 用）
            threshold_range: 阈值列表（threshold_sweep 用），默认 [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            parent_id: 父实验 ID（全量实验的子实验用）
        """
        exp_type = ExperimentType(experiment_type)

        if exp_type == ExperimentType.COMPONENT_ABLATION:
            engines = [
                "simple_minicpm_full",
                "simple_minicpm_no_simple_gate",
                "simple_minicpm_no_cooldown",
                "minicpm_full",
                "simple_full",
            ]
        elif exp_type == ExperimentType.THRESHOLD_SWEEP:
            engines = engine_names or ["simple_minicpm", "minicpm"]
        else:
            default_engines = [
                "simple",
                "simple_minicpm",
                "minicpm",
            ]
            engines = engine_names or default_engines

        exp = self.storage.create_experiment(recording_id, engines)
        exp.experiment_type = exp_type
        if parent_id:
            exp.parent_id = parent_id
        self.storage.update_experiment(exp)

        async with self._lock:
            self._queue.append((exp, engines, exp_type, threshold_range))
            if self._queue_task is None or self._queue_task.done():
                self._queue_task = asyncio.create_task(self._process_queue())

        return exp

    async def run_full_suite(
        self,
        positive_recording_ids: List[str],
        negative_recording_ids: List[str],
    ) -> AblationExperiment:
        """一键全量实验：对正样本和负样本分别运行实验，最后合并分析。

        必须至少提供一个正样本（包含 waving）和一个负样本（不包含 waving），
        以便分别评估召回率与误检率。
        """
        if not positive_recording_ids:
            raise ValueError("全量实验至少需要 1 个正样本录制")
        if not negative_recording_ids:
            raise ValueError("全量实验至少需要 1 个负样本录制")

        # 验证录制存在
        for rid in positive_recording_ids + negative_recording_ids:
            rec = self.storage.get_recording(rid)
            if rec is None:
                raise ValueError(f"录制不存在: {rid}")

        # 创建父实验（recording_id 使用第一个正样本作为代表）
        parent = self.storage.create_experiment(positive_recording_ids[0], [])
        parent.experiment_type = ExperimentType.FULL_SUITE
        parent.status = ExperimentStatus.RUNNING
        parent.positive_recording_ids = positive_recording_ids
        parent.negative_recording_ids = negative_recording_ids
        self.storage.update_experiment(parent)

        children: List[AblationExperiment] = []

        # ---- 正样本子实验 ----
        # 1. 引擎横向对比
        pos_ec = await self.run_experiment(
            positive_recording_ids[0], "engine_comparison", parent_id=parent.id
        )
        pos_ec.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ec)
        children.append(pos_ec)

        # 2. 组件消融
        pos_ca = await self.run_experiment(
            positive_recording_ids[0], "component_ablation", parent_id=parent.id
        )
        pos_ca.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ca)
        children.append(pos_ca)

        # 3. 阈值扫描
        pos_ts = await self.run_experiment(
            positive_recording_ids[0],
            "threshold_sweep",
            engine_names=["simple_minicpm", "minicpm"],
            parent_id=parent.id,
        )
        pos_ts.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ts)
        children.append(pos_ts)

        # 4. 场景分析
        pos_sa = await self.run_experiment(
            positive_recording_ids[0], "scenario_analysis", parent_id=parent.id
        )
        pos_sa.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_sa)
        children.append(pos_sa)

        # ---- 负样本子实验 ----
        # 仅运行引擎横向对比（核心误检评估）
        neg_ec = await self.run_experiment(
            negative_recording_ids[0], "engine_comparison", parent_id=parent.id
        )
        neg_ec.negative_recording_ids = negative_recording_ids
        self.storage.update_experiment(neg_ec)
        children.append(neg_ec)

        parent.sub_experiment_ids = [c.id for c in children]
        self.storage.update_experiment(parent)

        # 启动父实验 watcher
        asyncio.create_task(self._watch_parent(parent.id, parent.sub_experiment_ids))

        return parent

    async def _watch_parent(self, parent_id: str, child_ids: List[str]) -> None:
        """轮询子实验状态，更新父实验进度和最终状态。"""
        while True:
            await asyncio.sleep(5.0)
            parent = self.storage.get_experiment(parent_id)
            if not parent or parent.status != ExperimentStatus.RUNNING:
                break

            children = [self.storage.get_experiment(cid) for cid in child_ids]
            valid_children = [c for c in children if c]
            if not valid_children:
                break

            # 进度 = 已完成子实验 / 总数
            completed = sum(
                1 for c in valid_children if c.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
            )
            parent.progress = round(completed / len(valid_children), 4)

            # 任一失败则父实验失败
            if any(c.status == ExperimentStatus.FAILED for c in valid_children):
                parent.status = ExperimentStatus.FAILED
                parent.error_message = "部分子实验执行失败"
                self.storage.update_experiment(parent)
                break

            # 全部完成
            if completed == len(valid_children):
                parent.status = ExperimentStatus.COMPLETED
                parent.completed_at = time.time()
                parent.progress = 1.0
                self.storage.update_experiment(parent)
                logger.info("全量实验完成: %s", parent_id)
                # 生成合并报告
                try:
                    from app.datalab.analyzer import AblationAnalyzer
                    analyzer = AblationAnalyzer(self.storage)
                    analyzer.analyze_full_suite(parent_id)
                except Exception as e:
                    logger.error("全量实验合并报告生成失败: %s", e, exc_info=True)
                break

            self.storage.update_experiment(parent)

    async def _process_queue(self) -> None:
        """队列处理器：串行执行排队实验。"""
        while True:
            async with self._lock:
                if not self._queue:
                    self._current_exp = None
                    break
                exp, engines, exp_type, threshold_range = self._queue.pop(0)
                self._current_exp = exp
                self._cancelled = False

            # 在独立线程中运行实验，避免阻塞主事件循环的 HTTP 处理
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self._run_background_in_thread, exp, engines, exp_type, threshold_range
            )

    def _run_background_in_thread(
        self,
        exp: AblationExperiment,
        engine_names: List[str],
        exp_type: ExperimentType,
        threshold_range: Optional[List[float]] = None,
    ) -> None:
        """在线程中运行实验，创建独立的事件循环避免阻塞主循环。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._run_background(exp, engine_names, exp_type, threshold_range)
            )
        finally:
            loop.close()

    async def _run_background(
        self,
        exp: AblationExperiment,
        engine_names: List[str],
        exp_type: ExperimentType,
        threshold_range: Optional[List[float]] = None,
    ) -> None:
        """后台运行实验主体。支持单录制或多录制拼接（全量实验正负样本）。"""
        try:
            exp.status = ExperimentStatus.RUNNING
            self.storage.update_experiment(exp)

            # 决定使用哪些录制 ID（支持全量实验的多录制拼接）
            recording_ids: List[str] = []
            is_negative = False
            if exp.positive_recording_ids:
                recording_ids = exp.positive_recording_ids
            elif exp.negative_recording_ids:
                recording_ids = exp.negative_recording_ids
                is_negative = True
            else:
                recording_ids = [exp.recording_id]

            keypoints_frames, tnlf_frames, detections_frames = self._load_combined_frames(
                recording_ids, is_negative=is_negative
            )

            total_frames = len(keypoints_frames)
            if total_frames == 0:
                raise ValueError("录制数据为空")

            exp.total_frames = total_frames
            self.storage.update_experiment(exp)

            # 打开视频文件（供 MiniCPM 引擎使用真实帧）
            self._current_video_cap = self._open_recording_video(recording_ids)

            if exp_type == ExperimentType.THRESHOLD_SWEEP:
                await self._run_threshold_sweep(exp, keypoints_frames, tnlf_frames, engine_names, threshold_range, detections_frames)
            elif exp_type == ExperimentType.COMPONENT_ABLATION:
                await self._run_component_ablation(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)
            elif exp_type == ExperimentType.SCENARIO_ANALYSIS:
                await self._run_scenario_analysis(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)
            else:
                await self._run_standard_comparison(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)

            exp.status = ExperimentStatus.COMPLETED
            exp.completed_at = time.time()
            exp.progress = 1.0
            self.storage.update_experiment(exp)
            logger.info("消融实验完成: %s (type=%s)", exp.id, exp_type.value)

            # 若存在父实验，立即触发一次父状态更新
            if getattr(exp, 'parent_id', None):
                await self._check_parent_status(exp.parent_id)

        except Exception as e:
            logger.error("消融实验失败: %s", e, exc_info=True)
            exp.status = ExperimentStatus.FAILED
            exp.error_message = str(e)
            self.storage.update_experiment(exp)
            if getattr(exp, 'parent_id', None):
                await self._check_parent_status(exp.parent_id)
        finally:
            if getattr(self, '_current_video_cap', None) is not None:
                self._current_video_cap.release()
                self._current_video_cap = None

    async def _check_parent_status(self, parent_id: str) -> None:
        """检查父实验状态（由子实验完成后立即触发）。"""
        parent = self.storage.get_experiment(parent_id)
        if not parent or parent.status != ExperimentStatus.RUNNING:
            return
        child_ids = parent.sub_experiment_ids
        if not child_ids:
            return
        children = [self.storage.get_experiment(cid) for cid in child_ids]
        valid_children = [c for c in children if c]
        if not valid_children:
            return
        completed = sum(
            1 for c in valid_children if c.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
        )
        parent.progress = round(completed / len(valid_children), 4)
        if any(c.status == ExperimentStatus.FAILED for c in valid_children):
            parent.status = ExperimentStatus.FAILED
            parent.error_message = "部分子实验执行失败"
        elif completed == len(valid_children):
            parent.status = ExperimentStatus.COMPLETED
            parent.completed_at = time.time()
            parent.progress = 1.0
            try:
                from app.datalab.analyzer import AblationAnalyzer
                analyzer = AblationAnalyzer(self.storage)
                analyzer.analyze_full_suite(parent_id)
            except Exception as e:
                logger.error("全量实验合并报告生成失败: %s", e, exc_info=True)
        self.storage.update_experiment(parent)

    async def cancel(self) -> None:
        """取消当前实验并清空队列。"""
        self._cancelled = True
        async with self._lock:
            self._queue.clear()

    # ------------------------------------------------------------------
    # 标准引擎对比
    # ------------------------------------------------------------------

    async def _run_standard_comparison(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_engines(engine_names)
        logger.info("标准引擎对比: exp=%s engines=%s frames=%d", exp.id, engine_names, len(keypoints_frames))
        await self._infer_all_frames(exp, keypoints_frames, tnlf_frames, engines, detections_frames)

    # ------------------------------------------------------------------
    # 组件消融
    # ------------------------------------------------------------------

    async def _run_component_ablation(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_ablation_engines(engine_names)
        logger.info("组件消融: exp=%s variants=%s frames=%d", exp.id, list(engines.keys()), len(keypoints_frames))
        await self._infer_all_frames(exp, keypoints_frames, tnlf_frames, engines, detections_frames)

    # ------------------------------------------------------------------
    # 阈值扫描
    # ------------------------------------------------------------------

    async def _run_threshold_sweep(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        threshold_range: Optional[List[float]],
        detections_frames: List[Dict],
    ) -> None:
        # 更细粒度阈值扫描（0.05~0.95，步长0.05），确保曲线覆盖完整区间
        if threshold_range:
            thresholds = threshold_range
        else:
            thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05).tolist()]
        total = len(keypoints_frames)
        processed = 0

        for engine_name in engine_names:
            for thr in thresholds:
                if self._cancelled:
                    return

                engines = {engine_name: self._instantiate_engine_with_threshold(engine_name, thr)}

                for idx, kp_frame in enumerate(keypoints_frames):
                    if self._cancelled:
                        return
                    row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)
                    row["threshold"] = thr
                    self.storage.append_frame_result(exp.id, row)

                processed += total
                exp.current_frame = min(processed, total * len(engine_names) * len(thresholds))
                exp.progress = round(processed / (total * len(engine_names) * len(thresholds)), 4)
                self.storage.update_experiment(exp)
                await asyncio.sleep(0)

                # 等待 MiniCPM 异步推理完成并回填结果
                await self._backfill_minicpm_results(exp, engines, keypoints_frames)

                # 释放引擎状态
                engines[engine_name].reset()

    # ------------------------------------------------------------------
    # 场景分析
    # ------------------------------------------------------------------

    async def _run_scenario_analysis(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_engines(engine_names)
        logger.info("场景分析: exp=%s engines=%s frames=%d", exp.id, engine_names, len(keypoints_frames))

        for idx, kp_frame in enumerate(keypoints_frames):
            if self._cancelled:
                return

            tnlf = tnlf_frames[idx] if idx < len(tnlf_frames) else {}
            row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)

            # 计算场景标签
            v_left = float(tnlf.get("left_velocity_mag", 0.0))
            v_right = float(tnlf.get("right_velocity_mag", 0.0))
            v_max = max(v_left, v_right)

            if v_max < 0.03:
                row["scenario_velocity"] = "static"
            elif v_max < 0.1:
                row["scenario_velocity"] = "slow"
            else:
                row["scenario_velocity"] = "fast"

            left_valid = bool(tnlf.get("left_tnlf_valid", False))
            right_valid = bool(tnlf.get("right_tnlf_valid", False))
            if left_valid and right_valid:
                row["scenario_hand"] = "both"
            elif left_valid:
                row["scenario_hand"] = "left"
            elif right_valid:
                row["scenario_hand"] = "right"
            else:
                row["scenario_hand"] = "none"

            keypoints = np.array(kp_frame.get("keypoints", []))
            if keypoints.ndim == 1:
                keypoints = keypoints.reshape(-1, 3)
            if len(keypoints) >= 12:
                # 肩宽作为距离代理
                shoulder_dist = float(np.linalg.norm(keypoints[5, :2] - keypoints[6, :2]))
                if shoulder_dist < 80:
                    row["scenario_distance"] = "far"
                elif shoulder_dist < 150:
                    row["scenario_distance"] = "mid"
                else:
                    row["scenario_distance"] = "near"
            else:
                row["scenario_distance"] = "unknown"

            self.storage.append_frame_result(exp.id, row)

            exp.current_frame = idx + 1
            exp.progress = round((idx + 1) / len(keypoints_frames), 4)
            if idx % 30 == 0:
                self.storage.update_experiment(exp)
            if idx % 5 == 0:
                await asyncio.sleep(0)

        # 等待 MiniCPM 异步推理完成并回填结果
        await self._backfill_minicpm_results(exp, engines, keypoints_frames)

    # ------------------------------------------------------------------
    # 通用推理
    # ------------------------------------------------------------------

    async def _infer_all_frames(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engines: Dict[str, Any],
        detections_frames: List[Dict],
    ) -> None:
        for idx, kp_frame in enumerate(keypoints_frames):
            if self._cancelled:
                return

            row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)
            self.storage.append_frame_result(exp.id, row)

            exp.current_frame = idx + 1
            exp.progress = round((idx + 1) / len(keypoints_frames), 4)
            if idx % 30 == 0:
                self.storage.update_experiment(exp)
            if idx % 5 == 0:
                await asyncio.sleep(0)

        # 等待 MiniCPM 异步推理完成并回填结果
        await self._backfill_minicpm_results(exp, engines, keypoints_frames)

    async def _backfill_minicpm_results(
        self, exp: AblationExperiment,
        engines: Dict[str, Any],
        keypoints_frames: List[Dict],
    ) -> None:
        """等待 MiniCPM 引擎异步推理完成，按精确窗口回填每个 frame 的结果。

        每个 CHECKING 帧查询其所属的 MiniCPM 推理窗口，使用窗口自身的 gesture/conf 回填。
        未被任何窗口覆盖的 CHECKING 帧标记为 'none'（Simple trigger 通过了但 MiniCPM 从未分析）。
        """
        from app.ai.gesture import SimpleMiniCPMHybridRecognizer, MiniCPMGestureRecognizer

        # 1. 等待每个 MiniCPM 引擎的结果并收集推理窗口（在线程池中执行，避免阻塞事件循环）
        engine_windows: Dict[str, List[Tuple[int, int, str, float]]] = {}
        loop = asyncio.get_running_loop()
        for name, engine in engines.items():
            if isinstance(engine, (SimpleMiniCPMHybridRecognizer, MiniCPMGestureRecognizer)):
                track_id = keypoints_frames[-1].get("track_id", "person_1") if keypoints_frames else "person_1"
                logger.info("等待 MiniCPM 引擎 %s 异步结果 (track_id=%s)...", name, track_id)
                gesture, conf = await loop.run_in_executor(None, engine.wait_for_result, track_id, 60.0)
                logger.info("MiniCPM 引擎 %s 最终结果: %s %.2f", name, gesture, conf)
                windows = engine.get_inference_windows(track_id)
                engine_windows[name] = windows
                logger.info("MiniCPM 引擎 %s 推理窗口: %s", name, windows)

        if not engine_windows:
            return

        # 2. 读取现有 frame_results
        rows = self.storage.read_frame_results(exp.id)
        if not rows:
            return

        # 3. 按引擎处理：每个 CHECKING 帧精确匹配到所属窗口，使用该窗口的结果回填
        modified = False
        for name, windows in engine_windows.items():
            g_key = f"{name}_gesture"
            c_key = f"{name}_confidence"

            for row in rows:
                if row.get(g_key) != "checking":
                    continue
                row_frame_idx = row.get("frame_idx", 0)

                # 查找覆盖该帧的推理窗口（使用窗口自身的结果）
                matched_gesture = "none"
                matched_conf = 0.0
                for start_idx, end_idx, gesture, conf in windows:
                    if start_idx <= row_frame_idx <= end_idx:
                        matched_gesture = gesture
                        matched_conf = conf
                        break

                row[g_key] = matched_gesture
                row[c_key] = round(matched_conf, 4)
                modified = True

        # 4. 保存各引擎的 MiniCPM 推理次数（用于后续效率评分）
        inference_counts: Dict[str, int] = {}
        for name, windows in engine_windows.items():
            inference_counts[name] = len(windows)
        if inference_counts:
            exp_obj = self.storage.get_experiment(exp.id)
            if exp_obj:
                exp_dir = Path(exp_obj.frame_results_path).parent if exp_obj.frame_results_path else Path(f"data/datalab/experiments/{exp.id}")
                exp_dir.mkdir(parents=True, exist_ok=True)
                counts_path = exp_dir / "minicpm_inference_counts.json"
                with open(counts_path, "w", encoding="utf-8") as f:
                    json.dump(inference_counts, f, ensure_ascii=False, indent=2)
                logger.info("已保存 MiniCPM 推理次数: %s", inference_counts)

        if not modified:
            return

        # 5. 重写 frame_results.jsonl
        exp_obj = self.storage.get_experiment(exp.id)
        if exp_obj and exp_obj.frame_results_path:
            path = Path(exp_obj.frame_results_path)
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("已按精确推理窗口回填 MiniCPM 结果: %d 条 frame_results", len(rows))

    def _infer_single_frame(
        self, kp_frame: Dict, tnlf_frames: List[Dict], idx: int, engines: Dict[str, Any],
        detections_frames: List[Dict],
    ) -> Dict[str, Any]:
        frame_idx = kp_frame.get("frame_idx", idx)
        timestamp = kp_frame.get("timestamp", 0.0)
        keypoints = np.array(kp_frame.get("keypoints", []))
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 3)

        tnlf = tnlf_frames[idx] if idx < len(tnlf_frames) else {}
        left_wl = _to_array(tnlf.get("left_wrist_local"))
        right_wl = _to_array(tnlf.get("right_wrist_local"))
        left_pn = _to_array(kp_frame.get("left_palm_normal"))
        right_pn = _to_array(kp_frame.get("right_palm_normal"))

        kwargs = {
            "keypoints": keypoints,
            "track_id": kp_frame.get("track_id", "person_1"),
            "left_palm_normal": left_pn,
            "right_palm_normal": right_pn,
            "frame_timestamp": timestamp,
            "active_track_ids": {kp_frame.get("track_id", "person_1")},
            "left_wrist_local": left_wl,
            "right_wrist_local": right_wl,
            "left_tnlf_valid": tnlf.get("left_tnlf_valid", False),
            "right_tnlf_valid": tnlf.get("right_tnlf_valid", False),
            "left_velocity_mag": tnlf.get("left_velocity_mag", 0.0),
            "right_velocity_mag": tnlf.get("right_velocity_mag", 0.0),
            "left_theta1": tnlf.get("left_theta1", 0.0),
            "left_theta2": tnlf.get("left_theta2", 0.0),
            "left_ext_ratio": tnlf.get("left_ext_ratio", 0.0),
            "right_theta1": tnlf.get("right_theta1", 0.0),
            "right_theta2": tnlf.get("right_theta2", 0.0),
            "right_ext_ratio": tnlf.get("right_ext_ratio", 0.0),
        }

        # 为 MiniCPM 引擎准备 frame / bbox
        frame: Optional[np.ndarray] = None
        bbox: Optional[Tuple[int, int, int, int]] = None
        needs_frame_engines = [
            (name, engine) for name, engine in engines.items()
            if self._needs_frame(engine)
        ]
        if needs_frame_engines:
            # 1. 获取 bbox：优先用录制时保存的，否则从 keypoints 推断
            saved_bbox = kp_frame.get("bbox")
            if saved_bbox is not None and len(saved_bbox) == 4:
                bbox = tuple(int(v) for v in saved_bbox)
            else:
                bbox = self._bbox_from_keypoints(keypoints)

            # 2. 从视频读取对应帧
            cap = getattr(self, "_current_video_cap", None)
            if cap is not None and cap.isOpened() and bbox is not None:
                # frame_idx 是 1-based，set 到 0-based 位置
                target_pos = max(0, frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("无法读取视频帧 %d", frame_idx)
                    frame = None

        row: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        }
        for name, engine in engines.items():
            t0 = time.perf_counter()
            try:
                engine_kwargs = dict(kwargs)
                if self._needs_frame(engine):
                    engine_kwargs["frame"] = frame
                    engine_kwargs["bbox"] = bbox
                    engine_kwargs["frame_idx"] = frame_idx
                result = engine.recognize(**engine_kwargs)
                gesture = (
                    result.gesture_type.value
                    if hasattr(result.gesture_type, "value")
                    else str(result.gesture_type)
                )
                conf = result.confidence
            except Exception as e:
                logger.warning("引擎 %s 推理失败 frame=%d: %s", name, frame_idx, e)
                gesture = "error"
                conf = 0.0
            latency = (time.perf_counter() - t0) * 1000
            row[f"{name}_gesture"] = gesture
            row[f"{name}_confidence"] = round(max(0.0, min(1.0, conf)), 4)
            row[f"{name}_latency_ms"] = round(latency, 3)

        row["velocity_left"] = round(kwargs["left_velocity_mag"], 4)
        row["velocity_right"] = round(kwargs["right_velocity_mag"], 4)

        # 携带 ground truth（来自录制时的生产引擎检测结果）
        det = detections_frames[idx] if idx < len(detections_frames) else {}
        row["gt_gesture"] = det.get("gesture", "none")
        row["gt_conf"] = det.get("gesture_conf", 0.0)
        row["recording_id"] = det.get("_recording_id", "")
        return row

    # ------------------------------------------------------------------
    # 多录制数据拼接（全量实验正负样本支持）
    # ------------------------------------------------------------------

    def _load_combined_frames(
        self, recording_ids: List[str], is_negative: bool = False
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """加载并拼接多个录制的 keypoints、tnlf、detections 数据。

        对于负样本，强制将所有 detections 的 gesture 设为 'none'，
        因为负样本的 ground truth 就是无 waving。
        """
        keypoints_frames: List[Dict] = []
        tnlf_frames: List[Dict] = []
        detections_frames: List[Dict] = []

        for rid in recording_ids:
            kp_list = list(self.storage.iter_keypoints(rid))
            tnlf_list = list(self.storage.iter_tnlf(rid))
            det_list = list(self.storage.iter_detections(rid))

            # 统一长度（以 keypoints 为准）
            n = len(kp_list)
            if n == 0:
                logger.warning("录制 %s 无 keypoints 数据，跳过", rid)
                continue

            for i in range(n):
                kp = kp_list[i]
                tnlf = tnlf_list[i] if i < len(tnlf_list) else {}
                det = det_list[i] if i < len(det_list) else {}

                # 标记来源录制 ID，便于分析时区分
                det = dict(det)
                det["_recording_id"] = rid

                if is_negative:
                    # 负样本 ground truth 强制为 none
                    det["gesture"] = "none"
                    det["gesture_conf"] = 0.0
                else:
                    # 正样本：所有有人出现的帧都应当作 waving ground truth。
                    # 原 detections 中的 gesture 来自录制时的 Simple 引擎检测，
                    # 会漏掉大量实际 waving 帧，导致 MiniCPM 正确检测的帧被算作 FP。
                    # 修正：以 keypoints 存在性判断人物可见，标记为 waving。
                    keypoints = np.array(kp.get("keypoints", []))
                    if keypoints.ndim == 1:
                        keypoints = keypoints.reshape(-1, 3)
                    person_visible = (
                        len(keypoints) > 0
                        and np.any(keypoints[:, 2] > 0.15)
                    )
                    if person_visible:
                        det["gesture"] = "waving"
                        det["gesture_conf"] = 1.0
                    else:
                        det["gesture"] = "none"
                        det["gesture_conf"] = 0.0

                keypoints_frames.append(kp)
                tnlf_frames.append(tnlf)
                detections_frames.append(det)

        return keypoints_frames, tnlf_frames, detections_frames

    def _open_recording_video(self, recording_ids: List[str]) -> Optional[Any]:
        """打开第一个可用的录制视频文件，返回 cv2.VideoCapture。"""
        import cv2
        for rid in recording_ids:
            rec = self.storage.get_recording(rid)
            if rec and rec.video_path and os.path.exists(rec.video_path):
                cap = cv2.VideoCapture(rec.video_path)
                if cap.isOpened():
                    logger.info("实验使用录制视频: %s", rec.video_path)
                    return cap
                cap.release()
        logger.warning("无可用录制视频，MiniCPM 引擎将无法获得真实帧")
        return None

    @staticmethod
    def _bbox_from_keypoints(keypoints: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """从关键点推断人物 bbox（兼容旧数据无 bbox 的情况）。"""
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 3)
        if len(keypoints) == 0:
            return None
        # 取所有置信度 > 0.1 的关键点
        valid = keypoints[keypoints[:, 2] > 0.1]
        if len(valid) == 0:
            valid = keypoints
        xs = valid[:, 0]
        ys = valid[:, 1]
        margin = 20
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = int(xs.max()) + margin
        y2 = int(ys.max()) + margin
        return (x1, y1, x2, y2)

    @staticmethod
    def _needs_frame(engine: Any) -> bool:
        """判断引擎是否需要 frame/bbox 参数（MiniCPM 系列）。"""
        from app.ai.gesture import SimpleMiniCPMHybridRecognizer, MiniCPMGestureRecognizer
        return isinstance(engine, (SimpleMiniCPMHybridRecognizer, MiniCPMGestureRecognizer))

    def _get_shared_minicpm_engine(self) -> Any:
        """获取共享的 MiniCPMEngine 实例（单例，模型只加载一次，避免多实例并发加载导致模型损坏）。"""
        if self._shared_minicpm_engine is None:
            from app.ai.minicpm_engine import MiniCPMEngine
            minicpm_model = getattr(self.config.ai, "minicpm_model_path", "OpenBMB/MiniCPM-V-4.6")
            minicpm_prompt = getattr(self.config.ai, "minicpm_prompt", None)
            self._shared_minicpm_engine = MiniCPMEngine(
                model_path=minicpm_model,
                max_concurrent=16,
                prompt=minicpm_prompt,
            )
            logger.info("共享 MiniCPMEngine 已创建: %s", minicpm_model)
        return self._shared_minicpm_engine

    def _create_minicpm_engine(self) -> Any:
        """创建独立的 MiniCPMEngine 实例，但共享已加载的模型权重（避免并发加载损坏）。"""
        shared = self._get_shared_minicpm_engine()
        # 等待共享引擎模型加载完成
        import time
        for _ in range(300):
            if shared._loaded:
                break
            time.sleep(0.1)
        if not shared._loaded:
            raise RuntimeError("共享 MiniCPM 模型加载超时")
        from app.ai.minicpm_engine import MiniCPMEngine
        minicpm_model = getattr(self.config.ai, "minicpm_model_path", "OpenBMB/MiniCPM-V-4.6")
        minicpm_prompt = getattr(self.config.ai, "minicpm_prompt", None)
        return MiniCPMEngine(
            model_path=minicpm_model,
            max_concurrent=16,
            prompt=minicpm_prompt,
            model=shared._model,
            processor=shared._processor,
        )

    # ------------------------------------------------------------------
    # 引擎工厂
    # ------------------------------------------------------------------

    def _instantiate_engines(self, engine_names: List[str]) -> Dict[str, Any]:
        """实例化标准引擎。"""
        engines: Dict[str, Any] = {}
        for name in engine_names:
            engines[name] = self._instantiate_engine(name)
        return engines

    def _instantiate_engine(self, name: str) -> Any:
        from app.ai.gesture import (
            SimpleGestureRecognizer,
            SimpleMiniCPMHybridRecognizer,
            MiniCPMGestureRecognizer,
        )

        minicpm_model = getattr(self.config.ai, "minicpm_model_path", "OpenBMB/MiniCPM-V-4.6")
        minicpm_conf = getattr(self.config.ai, "minicpm_confidence_threshold", 0.6)
        minicpm_prompt = getattr(self.config.ai, "minicpm_prompt", None)
        minicpm_input_size = getattr(self.config.ai, "minicpm_input_size", 448)
        minicpm_max_frames = getattr(self.config.ai, "minicpm_max_frames", 16)
        minicpm_buffer_seconds = getattr(self.config.ai, "minicpm_buffer_seconds", 1.5)
        minicpm_min_frames = getattr(self.config.ai, "minicpm_min_frames", 8)
        minicpm_infer_interval_s = getattr(self.config.ai, "minicpm_infer_interval_s", 1.0)
        minicpm_max_concurrent = getattr(self.config.ai, "minicpm_max_concurrent", 16)
        minicpm_cooldown_seconds = getattr(self.config.ai, "minicpm_cooldown_seconds", 2.0)

        if name == "simple":
            # 消融实验中使用与 SimpleMiniCPM trigger 相同的宽松阈值，
            # 确保低质量视频下公平对比
            return SimpleGestureRecognizer(
                nose_conf_threshold=0.15,
                eye_conf_threshold=0.15,
                wrist_elbow_conf_threshold=0.10,
            )
        elif name == "simple_minicpm":
            return SimpleMiniCPMHybridRecognizer(
                model_path=minicpm_model,
                input_size=minicpm_input_size,
                max_frames=minicpm_max_frames,
                buffer_seconds=minicpm_buffer_seconds,
                min_frames=minicpm_min_frames,
                infer_interval_s=minicpm_infer_interval_s,
                max_concurrent=minicpm_max_concurrent,
                confidence_threshold=minicpm_conf,
                cooldown_seconds=minicpm_cooldown_seconds,
                prompt=minicpm_prompt,
                minicpm_engine=self._create_minicpm_engine(),
            )
        elif name == "minicpm":
            return MiniCPMGestureRecognizer(
                model_path=minicpm_model,
                input_size=minicpm_input_size,
                max_frames=minicpm_max_frames,
                buffer_seconds=minicpm_buffer_seconds,
                min_frames=minicpm_min_frames,
                infer_interval_s=minicpm_infer_interval_s,
                max_concurrent=minicpm_max_concurrent,
                confidence_threshold=minicpm_conf,
                cooldown_seconds=minicpm_cooldown_seconds,
                prompt=minicpm_prompt,
                minicpm_engine=self._create_minicpm_engine(),
            )
        else:
            raise ValueError(f"未知引擎: {name}")

    def _instantiate_engine_with_threshold(self, name: str, threshold: float) -> Any:
        from app.ai.gesture import (
            SimpleMiniCPMHybridRecognizer,
            MiniCPMGestureRecognizer,
        )

        minicpm_model = getattr(self.config.ai, "minicpm_model_path", "OpenBMB/MiniCPM-V-4.6")
        minicpm_prompt = getattr(self.config.ai, "minicpm_prompt", None)
        minicpm_input_size = getattr(self.config.ai, "minicpm_input_size", 448)
        minicpm_max_frames = getattr(self.config.ai, "minicpm_max_frames", 16)
        minicpm_buffer_seconds = getattr(self.config.ai, "minicpm_buffer_seconds", 1.5)
        minicpm_min_frames = getattr(self.config.ai, "minicpm_min_frames", 8)
        minicpm_infer_interval_s = getattr(self.config.ai, "minicpm_infer_interval_s", 1.0)
        minicpm_max_concurrent = getattr(self.config.ai, "minicpm_max_concurrent", 16)
        minicpm_cooldown_seconds = getattr(self.config.ai, "minicpm_cooldown_seconds", 2.0)

        if name == "simple_minicpm":
            return SimpleMiniCPMHybridRecognizer(
                model_path=minicpm_model,
                input_size=minicpm_input_size,
                max_frames=minicpm_max_frames,
                buffer_seconds=minicpm_buffer_seconds,
                min_frames=minicpm_min_frames,
                infer_interval_s=minicpm_infer_interval_s,
                max_concurrent=minicpm_max_concurrent,
                confidence_threshold=threshold,
                cooldown_seconds=minicpm_cooldown_seconds,
                prompt=minicpm_prompt,
                minicpm_engine=self._create_minicpm_engine(),
            )
        elif name == "minicpm":
            return MiniCPMGestureRecognizer(
                model_path=minicpm_model,
                input_size=minicpm_input_size,
                max_frames=minicpm_max_frames,
                buffer_seconds=minicpm_buffer_seconds,
                min_frames=minicpm_min_frames,
                infer_interval_s=minicpm_infer_interval_s,
                max_concurrent=minicpm_max_concurrent,
                confidence_threshold=threshold,
                cooldown_seconds=minicpm_cooldown_seconds,
                prompt=minicpm_prompt,
                minicpm_engine=self._create_minicpm_engine(),
            )
        else:
            return self._instantiate_engine(name)

    def _instantiate_ablation_engines(self, engine_names: List[str]) -> Dict[str, Any]:
        """实例化组件消融变体引擎。"""
        engines: Dict[str, Any] = {}
        minicpm_conf = getattr(self.config.ai, "minicpm_confidence_threshold", 0.6)
        minicpm_model = getattr(self.config.ai, "minicpm_model_path", "OpenBMB/MiniCPM-V-4.6")
        minicpm_prompt = getattr(self.config.ai, "minicpm_prompt", None)
        minicpm_input_size = getattr(self.config.ai, "minicpm_input_size", 448)
        minicpm_max_frames = getattr(self.config.ai, "minicpm_max_frames", 16)
        minicpm_buffer_seconds = getattr(self.config.ai, "minicpm_buffer_seconds", 1.5)
        minicpm_min_frames = getattr(self.config.ai, "minicpm_min_frames", 8)
        minicpm_infer_interval_s = getattr(self.config.ai, "minicpm_infer_interval_s", 1.0)
        minicpm_max_concurrent = getattr(self.config.ai, "minicpm_max_concurrent", 16)
        minicpm_cooldown_seconds = getattr(self.config.ai, "minicpm_cooldown_seconds", 2.0)

        from app.ai.gesture import (
            SimpleGestureRecognizer,
            SimpleMiniCPMHybridRecognizer,
            MiniCPMGestureRecognizer,
        )

        for name in engine_names:
            if name == "simple_minicpm_full":
                engines[name] = SimpleMiniCPMHybridRecognizer(
                    model_path=minicpm_model,
                    input_size=minicpm_input_size,
                    max_frames=minicpm_max_frames,
                    buffer_seconds=minicpm_buffer_seconds,
                    min_frames=minicpm_min_frames,
                    infer_interval_s=minicpm_infer_interval_s,
                    max_concurrent=minicpm_max_concurrent,
                    confidence_threshold=minicpm_conf,
                    cooldown_seconds=minicpm_cooldown_seconds,
                    prompt=minicpm_prompt,
                    minicpm_engine=self._create_minicpm_engine(),
                )
            elif name == "simple_minicpm_no_simple_gate":
                base = SimpleMiniCPMHybridRecognizer(
                    model_path=minicpm_model,
                    input_size=minicpm_input_size,
                    max_frames=minicpm_max_frames,
                    buffer_seconds=minicpm_buffer_seconds,
                    min_frames=minicpm_min_frames,
                    infer_interval_s=minicpm_infer_interval_s,
                    max_concurrent=minicpm_max_concurrent,
                    confidence_threshold=minicpm_conf,
                    cooldown_seconds=minicpm_cooldown_seconds,
                    prompt=minicpm_prompt,
                    minicpm_engine=self._create_minicpm_engine(),
                )
                engines[name] = _SimpleMiniCPMNoSimpleGate(base)
            elif name == "simple_minicpm_no_cooldown":
                base = SimpleMiniCPMHybridRecognizer(
                    model_path=minicpm_model,
                    input_size=minicpm_input_size,
                    max_frames=minicpm_max_frames,
                    buffer_seconds=minicpm_buffer_seconds,
                    min_frames=minicpm_min_frames,
                    infer_interval_s=minicpm_infer_interval_s,
                    max_concurrent=minicpm_max_concurrent,
                    confidence_threshold=minicpm_conf,
                    cooldown_seconds=minicpm_cooldown_seconds,
                    prompt=minicpm_prompt,
                    minicpm_engine=self._create_minicpm_engine(),
                )
                engines[name] = _SimpleMiniCPMNoCooldown(base)
            elif name == "minicpm_full":
                engines[name] = MiniCPMGestureRecognizer(
                    model_path=minicpm_model,
                    input_size=minicpm_input_size,
                    max_frames=minicpm_max_frames,
                    buffer_seconds=minicpm_buffer_seconds,
                    min_frames=minicpm_min_frames,
                    infer_interval_s=minicpm_infer_interval_s,
                    max_concurrent=minicpm_max_concurrent,
                    confidence_threshold=minicpm_conf,
                    cooldown_seconds=minicpm_cooldown_seconds,
                    prompt=minicpm_prompt,
                    minicpm_engine=self._create_minicpm_engine(),
                )
            elif name == "simple_full":
                engines[name] = SimpleGestureRecognizer(
                    nose_conf_threshold=0.15,
                    eye_conf_threshold=0.15,
                    wrist_elbow_conf_threshold=0.10,
                )
            else:
                logger.warning("未知消融变体: %s", name)
        return engines

    def get_progress(self) -> Optional[AblationExperiment]:
        """获取当前实验进度。"""
        return self._current_exp


def _to_array(val: Any) -> Optional[np.ndarray]:
    """将 list/tuple 转为 numpy array。"""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    try:
        return np.array(val, dtype=np.float32)
    except (ValueError, TypeError):
        return None
