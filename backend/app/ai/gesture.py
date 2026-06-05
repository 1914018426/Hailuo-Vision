"""
手势识别模块 —— 三锁合取机制 (Triple-Lock Conjunction)

删除 idle → posed → oscillating → confirmed 状态机，改为：
  姿态锁 + 朝向锁 + 运动锁 同时满足 → confirmed_hailing

所有轨迹、速度、周期性判断基于 Torso-Normalized Local Frame (TNLF)。
禁止直接使用画面像素坐标。
"""

import logging
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.config import get_config
from app.ai.local_frame import wrist_to_local_frame, local_velocity
from app.ai.facing import facing_gate
from app.ai.slerp import (
    compute_palm_normal,
    NormalSmoother,
    angle_to_camera_z,
    angle_to_camera_z_weighted,
)
from app.ai.iri import IRICalculator

logger = logging.getLogger(__name__)


# =============================================================================
# 周期性运动检测引擎（基于 wrist_local，单位 torso_units）
# =============================================================================

class PeriodicMotionDetector:
    """
    基于 wrist_local 序列的周期性运动检测器。

    使用 zero-crossing + 自相关函数(ACF) 检测稳定的周期性运动。
    人类挥手/招手的典型频率：0.5-3 Hz。
    """

    def __init__(
        self,
        buffer_seconds: float = 2.0,
        fps: float = 15.0,
        min_freq_hz: float = 0.35,
        max_freq_hz: float = 3.0,
        min_cycles: int = 2,
        max_cycle_variation: float = 0.45,
    ) -> None:
        self.fps = fps
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_cycles = min_cycles
        self.max_cycle_variation = max_cycle_variation
        self._maxlen = int(buffer_seconds * fps)
        self._xs: deque = deque(maxlen=self._maxlen)
        self._ys: deque = deque(maxlen=self._maxlen)
        self._last_result: Optional[Dict[str, Any]] = None
        self._dirty: bool = False
        self._cached_result: Optional[Dict[str, Any]] = None
        # 滑动平均窗口：抑制车辆颠簸导致的高频抖动
        self._x_win: deque = deque(maxlen=3)
        self._y_win: deque = deque(maxlen=3)

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()
        self._last_result = None
        self._dirty = False
        self._cached_result = None
        self._x_win.clear()
        self._y_win.clear()

    def feed(self, wrist_local: Tuple[float, float]) -> None:
        """喂入新一帧的 wrist_local（躯干归一化坐标），先做 3 帧滑动平均降噪。"""
        self._x_win.append(float(wrist_local[0]))
        self._y_win.append(float(wrist_local[1]))
        self._xs.append(sum(self._x_win) / len(self._x_win))
        self._ys.append(sum(self._y_win) / len(self._y_win))
        self._dirty = True

    def detect(self) -> Optional[Dict[str, Any]]:
        """
        检测周期性运动。结果在同一帧内缓存（dirty-flag），
        避免 MediaPipe 降采样跳帧时重复执行 FFT。

        Returns:
            dict with keys:
                - is_periodic: bool
                - dominant_axis: "x" | "y" | "none"
                - frequency_hz: float
                - amplitude_tu: float   # 振幅（已是 torso_units）
                - cycle_count: int
                - consistency: float    # 0-1，周期一致性
                - zero_crossings: int
        """
        if not self._dirty and self._cached_result is not None:
            return self._cached_result

        if len(self._xs) < self._maxlen * 0.6:
            self._cached_result = None
            self._dirty = False
            return None

        x_result = self._analyze_axis(np.array(self._xs))
        y_result = self._analyze_axis(np.array(self._ys))

        if x_result is None and y_result is None:
            self._last_result = None
            return None

        if x_result is None:
            best = y_result
            best["dominant_axis"] = "y"
        elif y_result is None:
            best = x_result
            best["dominant_axis"] = "x"
        elif x_result.get("consistency", 0) >= y_result.get("consistency", 0):
            best = x_result
            best["dominant_axis"] = "x"
        else:
            best = y_result
            best["dominant_axis"] = "y"

        self._last_result = best
        self._dirty = False
        self._cached_result = best
        return best

    def _analyze_axis(self, series: np.ndarray) -> Optional[Dict[str, Any]]:
        """分析单轴序列的周期性。"""
        if len(series) < 10:
            return None

        # 1. 去趋势
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        detrended = series - (slope * x + intercept)

        # 2. Zero-crossing 检测
        zero_crossings = self._count_zero_crossings(detrended)
        if zero_crossings < self.min_cycles * 2:
            return None

        # 3. 从 zero-crossing 估计频率
        duration_sec = len(series) / self.fps
        freq_zc = zero_crossings / (2 * duration_sec)
        if not (self.min_freq_hz <= freq_zc <= self.max_freq_hz):
            return None

        # 4. ACF 峰值检测
        acf_result = PeriodicMotionDetector._acf_peak_period(
            detrended, self.fps, self.min_freq_hz, self.max_freq_hz
        )
        if acf_result is None:
            return None
        acf_period_frames, acf_peak_val = acf_result

        freq_acf = self.fps / acf_period_frames if acf_period_frames > 0 else 0
        if not (self.min_freq_hz <= freq_acf <= self.max_freq_hz):
            return None

        # 5. 频率一致性
        if freq_zc > 0 and abs(freq_acf - freq_zc) / freq_zc > 0.4:
            return None

        # 6. 周期一致性
        cycle_lengths = self._extract_cycle_lengths(detrended)
        consistency = self._compute_consistency(cycle_lengths)
        if consistency < 0.35:
            return None

        # 7. 振幅（已是 torso_units）
        amplitude = float((np.max(detrended) - np.min(detrended)) / 2.0)

        cycle_count = len(cycle_lengths)

        return {
            "is_periodic": True,
            "frequency_hz": float((freq_zc + freq_acf) / 2.0),
            "amplitude_tu": amplitude,
            "cycle_count": cycle_count,
            "consistency": consistency,
            "zero_crossings": zero_crossings,
            "acf_peak": acf_peak_val,
        }

    @staticmethod
    def _count_zero_crossings(series: np.ndarray) -> int:
        signs = np.sign(series)
        diff = np.diff(signs)
        return int(np.sum(diff != 0))

    @staticmethod
    def _acf_peak_period(
        series: np.ndarray, fps: float, min_freq: float, max_freq: float
    ) -> Optional[Tuple[int, float]]:
        n = len(series)
        if n < 10:
            return None
        s = series - np.mean(series)
        fft_result = np.fft.fft(s, n=n * 2)
        acf = np.fft.ifft(fft_result * np.conjugate(fft_result)).real[:n]
        acf = acf / acf[0]
        min_lag = max(3, int(fps / max_freq) - 1)
        max_lag = min(n // 2, int(fps / min_freq) + 3)
        if max_lag <= min_lag:
            return None
        peak_idx = min_lag + int(np.argmax(acf[min_lag:max_lag]))
        peak_val = float(acf[peak_idx])
        if peak_val < 0.18:
            return None
        return peak_idx, peak_val

    @staticmethod
    def _extract_cycle_lengths(series: np.ndarray) -> List[float]:
        signs = np.sign(series)
        crossings = []
        for i in range(1, len(signs)):
            if signs[i] == 0:
                continue
            if signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                crossings.append(i)
        if len(crossings) < 3:
            return []
        return [crossings[i] - crossings[i - 1] for i in range(1, len(crossings))]

    @staticmethod
    def _compute_consistency(lengths: List[float]) -> float:
        if len(lengths) < 2:
            return 0.0
        mean_len = np.mean(lengths)
        if mean_len < 1e-6:
            return 0.0
        std_len = np.std(lengths)
        cv = std_len / mean_len
        return float(max(0.0, 1.0 - cv))


# =============================================================================
# 归一化姿态特征提取器（角度链）
# =============================================================================

class NormalizedPoseFeatures:
    """将原始 COCO 关键点转换为躯干归一化坐标系，并计算关节角度链。"""

    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12

    def __init__(self, keypoints: np.ndarray) -> None:
        self.kpts = keypoints
        self.torso_size = self._compute_torso_size()

    def _kp(self, idx: int) -> Optional[np.ndarray]:
        if self.kpts is None or len(self.kpts) <= idx:
            return None
        kp = self.kpts[idx]
        if len(kp) >= 3 and kp[2] < 0.3:
            return None
        return np.array(kp[:2], dtype=float)

    def _compute_torso_size(self) -> float:
        ls = self._kp(self.L_SHOULDER)
        rs = self._kp(self.R_SHOULDER)
        lh = self._kp(self.L_HIP)
        rh = self._kp(self.R_HIP)
        if ls is None or rs is None or lh is None or rh is None:
            if ls is not None and rs is not None:
                return float(np.linalg.norm(ls - rs))
            return 100.0
        d1 = np.linalg.norm(ls - lh)
        d2 = np.linalg.norm(ls - rh)
        d3 = np.linalg.norm(rs - lh)
        d4 = np.linalg.norm(rs - rh)
        return float((d1 + d2 + d3 + d4) / 4.0)

    @staticmethod
    def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_ang = np.dot(ba, bc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    def theta1(self, side: str) -> Optional[float]:
        if side == "left":
            hip = self._kp(self.L_HIP)
            shoulder = self._kp(self.L_SHOULDER)
            elbow = self._kp(self.L_ELBOW)
        else:
            hip = self._kp(self.R_HIP)
            shoulder = self._kp(self.R_SHOULDER)
            elbow = self._kp(self.R_ELBOW)
        if shoulder is None or elbow is None:
            return None
        if hip is None:
            hip = np.array([shoulder[0], shoulder[1] + self.torso_size])
        return self._angle_3pt(hip, shoulder, elbow)

    def theta2(self, side: str) -> Optional[float]:
        if side == "left":
            s, e, w = self._kp(self.L_SHOULDER), self._kp(self.L_ELBOW), self._kp(self.L_WRIST)
        else:
            s, e, w = self._kp(self.R_SHOULDER), self._kp(self.R_ELBOW), self._kp(self.R_WRIST)
        if s is None or e is None or w is None:
            return None
        return self._angle_3pt(s, e, w)

    def arm_extension_ratio(self, side: str) -> Optional[float]:
        if side == "left":
            s, e, w = self._kp(self.L_SHOULDER), self._kp(self.L_ELBOW), self._kp(self.L_WRIST)
        else:
            s, e, w = self._kp(self.R_SHOULDER), self._kp(self.R_ELBOW), self._kp(self.R_WRIST)
        if s is None or e is None or w is None:
            return None
        d_se = np.linalg.norm(s - e)
        d_ew = np.linalg.norm(e - w)
        d_sw = np.linalg.norm(s - w)
        if d_se + d_ew < 1e-6:
            return None
        return float(d_sw / (d_se + d_ew))



# =============================================================================
# 三锁合取引擎
# =============================================================================

@dataclass
class TripleLockState:
    """单侧三锁状态（每 track_id + side 一个实例）。"""

    # 锁计数器（连续满足帧数）
    pose_count: int = 0
    orientation_count: int = 0
    motion_count: int = 0

    # 释放计数器（连续不满足帧数）
    pose_release: int = 0
    orientation_release: int = 0
    motion_release: int = 0

    # 确认后的保持
    hold_frames: int = 0
    confirmed: bool = False
    peak_confidence: float = 0.0

    # 历史数据
    wrist_local_history: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=30))
    periodic_detector: PeriodicMotionDetector = field(default_factory=lambda: PeriodicMotionDetector())
    # 前臂角度周期性检测器（向量化鲁棒性增强）
    angle_detector: PeriodicMotionDetector = field(default_factory=lambda: PeriodicMotionDetector())
    normal_smoother: NormalSmoother = field(default_factory=lambda: NormalSmoother(alpha=0.3))
    iri_calculator: IRICalculator = field(default_factory=lambda: IRICalculator(window_size=15))

    # 缓存
    last_wrist_local: Optional[Tuple[float, float]] = None
    last_timestamp: Optional[float] = None
    smoothed_confidence: float = 0.0
    # EMA 平滑后的 wrist_local，抑制 YOLO 检测抖动
    ema_wrist_local: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.pose_count = 0
        self.orientation_count = 0
        self.motion_count = 0
        self.pose_release = 0
        self.orientation_release = 0
        self.motion_release = 0
        self.hold_frames = 0
        self.confirmed = False
        self.peak_confidence = 0.0
        self.wrist_local_history.clear()
        self.timestamp_history.clear()
        self.periodic_detector.reset()
        self.angle_detector.reset()
        self.normal_smoother.reset()
        self.iri_calculator.reset()
        self.last_wrist_local = None
        self.last_timestamp = None
        self.smoothed_confidence = 0.0
        self.ema_wrist_local = None


class TripleLockEngine:
    """
    三锁合取机制。

    锁表：
    ┌──────────┬─────────────────────────────────────┬──────────┬─────────────────┐
    │ 锁       │ 判定条件                            │ 最小持续 │ 释放条件        │
    ├──────────┼─────────────────────────────────────┼──────────┼─────────────────┤
    │ 姿态锁   │ θ1 > 25° 且 θ2 > 15° 且 ext>0.1    │ 3帧      │ 任一条件不满足  │
    │ 朝向锁   │ n_smooth 与 Z 轴夹角 < 45°          │ 5帧      │ 夹角 > 60°      │
    │ 运动锁   │ FFT 主导频 0.5~3Hz 且幅度 > 0.1     │ 3帧      │ 周期消失或速<0.05│
    └──────────┴─────────────────────────────────────┴──────────┴─────────────────┘
    """

    def __init__(self) -> None:
        self.config = get_config()
        c = self.config.ai

        # 姿态锁阈值
        self.theta1_min = c.gesture_theta1_hailing_min
        self.theta2_min = c.gesture_theta2_straight_min
        self.ext_min = c.gesture_arm_extension_min

        # 朝向锁阈值
        self.orientation_lock_angle = c.gesture_orientation_lock_angle
        self.orientation_release_angle = c.gesture_orientation_release_angle

        # 运动锁阈值（torso_units/s）
        self.motion_freq_min = c.gesture_motion_freq_min
        self.motion_freq_max = c.gesture_motion_freq_max
        self.motion_amp_min = c.gesture_motion_amp_min
        self.motion_speed_min = c.gesture_motion_speed_min

        # 锁持续帧数
        self.pose_min_frames = c.gesture_pose_min_frames
        self.orientation_min_frames = c.gesture_orientation_min_frames
        self.motion_min_frames = c.gesture_motion_min_frames

        # 确认后保持
        self.hold_max_frames = c.gesture_hold_max_frames

        # 置信度平滑
        self.ema_alpha = c.gesture_ema_alpha

        # 每 track_id_side 一个状态
        self._states: Dict[str, TripleLockState] = {}

    def _state_key(self, track_id: str, side: str) -> str:
        return f"{track_id}_{side}"

    def _get_state(self, track_id: str, side: str) -> TripleLockState:
        key = self._state_key(track_id, side)
        if key not in self._states:
            self._states[key] = TripleLockState()
        return self._states[key]

    def _clear_state(self, track_id: str, side: str) -> None:
        key = self._state_key(track_id, side)
        if key in self._states:
            self._states[key].reset()
            del self._states[key]

    def gc_states(self, active_track_ids: set) -> None:
        stale = [k for k in self._states if k.rsplit("_", 1)[0] not in active_track_ids]
        for k in stale:
            self._states[k].reset()
            del self._states[k]

    def reset(self) -> None:
        for s in self._states.values():
            s.reset()
        self._states.clear()

    def process_frame(
        self,
        keypoints: np.ndarray,
        side: str,
        track_id: str,
        palm_normal: Optional[np.ndarray],
        timestamp: float,
        f_human: float,
        f_human_multiplier: float,
    ) -> Tuple[str, float]:
        """
        处理单帧，返回 (gesture, confidence)。
        """
        state = self._get_state(track_id, side)

        # ---- 1. 局部参考系转换 ----
        wrist_local, torso_scale, frame_valid = wrist_to_local_frame(keypoints, side)
        if not frame_valid or wrist_local is None:
            # 跳过此帧但保留累积状态（车辆振动导致关键点偶发置信度下降）
            return "none", 0.0

        # EMA 平滑：抑制 YOLO 关键点检测帧间抖动
        EMA_ALPHA = 0.25
        if state.ema_wrist_local is not None:
            sx = EMA_ALPHA * wrist_local[0] + (1 - EMA_ALPHA) * state.ema_wrist_local[0]
            sy = EMA_ALPHA * wrist_local[1] + (1 - EMA_ALPHA) * state.ema_wrist_local[1]
            wrist_local = (sx, sy)
        state.ema_wrist_local = wrist_local

        # ---- 2. 速度计算（基于平滑后的 wrist_local） ----
        v_mag = 0.0
        if state.last_wrist_local is not None and state.last_timestamp is not None:
            dt = timestamp - state.last_timestamp
            if dt > 1e-6:
                _, _, v_mag = local_velocity(state.last_wrist_local, wrist_local, dt)

        state.last_wrist_local = wrist_local
        state.last_timestamp = timestamp

        # ---- 3. 喂入周期性检测器（位置 + 前臂角度） ----
        state.periodic_detector.feed(wrist_local)
        period_info = state.periodic_detector.detect()

        # ---- 3b. 向量化鲁棒性增强：检测 TNLF 极角的周期性 ----
        # 车辆移动时，wrist_local 的 x/y 可能有缓慢漂移，但极角 theta 的周期性更鲁棒
        angle_info = None
        theta = math.degrees(math.atan2(wrist_local[1], wrist_local[0]))
        state.angle_detector.feed((theta, 0.0))
        angle_info = state.angle_detector.detect()

        # ---- 4. 姿态锁判定 ----
        feat = NormalizedPoseFeatures(keypoints)
        theta1 = feat.theta1(side)
        theta2 = feat.theta2(side)
        ext_ratio = feat.arm_extension_ratio(side)

        pose_ok = False
        pose_score = 0.0
        if theta1 is not None and theta2 is not None and ext_ratio is not None:
            if theta1 > self.theta1_min and theta2 > self.theta2_min and ext_ratio > self.ext_min:
                pose_ok = True
                pose_score = min(1.0, 0.5 + (theta1 - self.theta1_min) / 120.0)
                pose_score = min(1.0, pose_score + (theta2 - self.theta2_min) / 120.0)

        if pose_ok:
            state.pose_count += 1
            state.pose_release = 0
        else:
            state.pose_release += 1
            if state.pose_release >= 2:  # 允许 1 帧抖动
                state.pose_count = 0

        pose_locked = state.pose_count >= self.pose_min_frames

        # ---- 5. 朝向锁判定 ----
        orientation_ok = False
        if palm_normal is not None:
            angle = angle_to_camera_z_weighted(palm_normal, z_weight=0.3)
            orientation_ok = angle < self.orientation_lock_angle
        else:
            # 无手掌法向量数据（如 MediaPipe Hands 关闭的录制），跳过朝向锁
            orientation_ok = True

        if orientation_ok:
            state.orientation_count += 1
            state.orientation_release = 0
        else:
            state.orientation_release += 1
            if state.orientation_release >= 2:
                state.orientation_count = 0

        orientation_locked = state.orientation_count >= self.orientation_min_frames

        # ---- 6. 运动锁判定（位置周期 或 前臂角度周期） ----
        motion_ok = False
        motion_score = 0.0
        best_info = None

        # A. 手腕位置周期性
        if period_info and period_info.get("is_periodic"):
            freq = period_info.get("frequency_hz", 0.0)
            amp = period_info.get("amplitude_tu", 0.0)
            consistency = period_info.get("consistency", 0.0)
            freq_ok = self.motion_freq_min <= freq <= self.motion_freq_max
            amp_ok = amp > self.motion_amp_min
            if freq_ok and amp_ok:
                motion_ok = True
                motion_score = min(1.0, amp / 0.3) * min(1.0, consistency)
                best_info = period_info

        # B. 前臂角度周期性（向量化鲁棒性增强）
        if not motion_ok and angle_info and angle_info.get("is_periodic"):
            freq = angle_info.get("frequency_hz", 0.0)
            amp_deg = angle_info.get("amplitude_tu", 0.0)
            consistency = angle_info.get("consistency", 0.0)
            freq_ok = self.motion_freq_min <= freq <= self.motion_freq_max
            # 前臂摆动角度阈值：8 度 ≈ 0.14 rad
            if freq_ok and amp_deg > 8.0:
                motion_ok = True
                motion_score = min(1.0, amp_deg / 30.0) * min(1.0, consistency)
                best_info = angle_info

        # 若速度过低也释放运动锁
        if v_mag < self.motion_speed_min:
            motion_ok = False

        if motion_ok:
            state.motion_count += 1
            state.motion_release = 0
        else:
            state.motion_release += 1
            # 车辆颠簸可能导致短暂检测中断，放宽释放容忍到 5 帧
            if state.motion_release >= 5:
                state.motion_count = 0

        motion_locked = state.motion_count >= self.motion_min_frames

        # ---- 7. 三锁合取 / 保持 ----
        all_locked = pose_locked and orientation_locked and motion_locked

        # ---- IRI（意图刚性指数）----
        iri_r = state.iri_calculator.feed(keypoints, side, palm_normal)

        if all_locked:
            state.confirmed = True
            state.hold_frames = self.hold_max_frames
            # 记录峰值置信度：S = Pose_score * R * Motion_score * F_human
            state.peak_confidence = pose_score * iri_r * motion_score * f_human_multiplier

        gesture = "none"
        raw_conf = 0.0

        if state.confirmed and state.hold_frames > 0:
            # 保持期内输出衰减中的峰值置信度
            decay = state.hold_frames / self.hold_max_frames
            raw_conf = state.peak_confidence * decay
            gesture = "waving"
            state.hold_frames -= 1
            # 若保持期耗尽，释放 confirmed
            if state.hold_frames <= 0:
                state.confirmed = False
        else:
            state.confirmed = False
            state.hold_frames = 0
            state.peak_confidence = 0.0

        # EMA 平滑
        state.smoothed_confidence = (
            self.ema_alpha * raw_conf + (1 - self.ema_alpha) * state.smoothed_confidence
        )

        logger.debug(
            "triple-lock[%s/%s] pose=%s|%d orient=%s|%d motion=%s|%d "
            "v=%.3fTU/s conf=%.3f f_human=%.2f",
            track_id, side,
            "Y" if pose_locked else "N", state.pose_count,
            "Y" if orientation_locked else "N", state.orientation_count,
            "Y" if motion_locked else "N", state.motion_count,
            v_mag, state.smoothed_confidence, f_human,
        )

        return gesture, state.smoothed_confidence


# =============================================================================
# 手势识别器主类（兼容旧接口）
# =============================================================================

class GestureRecognizer:
    """
    帧级手势识别器 —— 三锁合取机制。
    """

    L_WRIST = 9
    R_WRIST = 10

    def __init__(self) -> None:
        self.config = get_config()
        self.engine = TripleLockEngine()

        logger.info(
            "GestureRecognizer(三锁合取): pose_min=%d orient_min=%d motion_min=%d hold=%d",
            self.engine.pose_min_frames,
            self.engine.orientation_min_frames,
            self.engine.motion_min_frames,
            self.engine.hold_max_frames,
        )

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        **kwargs,
    ) -> "GestureResult":
        """
        帧级手势识别。
        左右手分别运行三锁引擎，取置信度最高者。
        """
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        if active_track_ids is not None:
            self.engine.gc_states(active_track_ids)

        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 面部过滤（零模型）
        c = self.config.ai
        f_human, is_hard_rejected, f_human_multiplier = facing_gate(
            keypoints,
            hard_threshold=c.gesture_facing_hard_threshold,
            soft_threshold=c.gesture_facing_soft_threshold,
        )
        if is_hard_rejected:
            logger.debug("facing-gate[%s] hard-rejected f_human=%.2f", track_id, f_human)
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        best_result: Optional[GestureResult] = None

        for side in ["right", "left"]:
            raw_normal = left_palm_normal if side == "left" else right_palm_normal

            # ---- MediaPipe 法向量平滑 ----
            palm_normal = None
            if raw_normal is not None:
                state = self.engine._get_state(track_id, side)
                palm_normal = state.normal_smoother.update(raw_normal)

            gesture, confidence = self.engine.process_frame(
                keypoints=keypoints,
                side=side,
                track_id=track_id,
                palm_normal=palm_normal,
                timestamp=now,
                f_human=f_human,
                f_human_multiplier=f_human_multiplier,
            )

            result = GestureResult(
                gesture_type=GestureType.WAVING if gesture == "waving" else GestureType.NONE,
                confidence=confidence,
            )

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result if best_result else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0
        )

    def reset(self) -> None:
        """重置所有状态。"""
        self.engine.reset()
        logger.info("手势识别器已重置")


class TransformerGestureRecognizer:
    """
    基于 TemporalKeypointTransformer 的手势识别器。

    使用训练好的 Transformer 模型替代人工规则三锁合取机制，
    直接从 TNLF 特征序列学习挥手模式。
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 confidence_threshold: float = 0.5,
                 ema_alpha: float = 0.35,
                 hold_frames: int = 15) -> None:
        from app.ai.transformer.engine import TransformerGestureEngine
        self.engine = TransformerGestureEngine(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            ema_alpha=ema_alpha,
            hold_frames=hold_frames,
        )
        self.config = get_config()
        logger.info(
            "TransformerGestureRecognizer: model=%s threshold=%.2f",
            model_path, confidence_threshold,
        )

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        left_tnlf_valid: bool = False,
        right_tnlf_valid: bool = False,
        left_velocity_mag: float = 0.0,
        right_velocity_mag: float = 0.0,
        left_theta1: float = 0.0, left_theta2: float = 0.0, left_ext_ratio: float = 0.0,
        right_theta1: float = 0.0, right_theta2: float = 0.0, right_ext_ratio: float = 0.0,
    ) -> "GestureResult":
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 面部过滤（硬拒绝在 detector 层处理，这里仅做硬拒绝快速返回）
        c = self.config.ai
        f_human, is_hard_rejected, _ = facing_gate(
            keypoints,
            hard_threshold=c.gesture_facing_hard_threshold,
            soft_threshold=c.gesture_facing_soft_threshold,
        )
        if is_hard_rejected:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        # GC stale buffers
        if active_track_ids:
            active_keys = {f"{tid}_{side}" for tid in active_track_ids
                          for side in ["left", "right"]}
            self.engine.cleanup_stale(active_keys)

        best_result: Optional[GestureResult] = None

        # 前臂方向向量 (Forearm Direction Vector, FDV)
        #   定义：normalize(wrist - elbow) in XY 平面，Z = 0.5
        #   物理意义：描述"手臂朝哪个方向伸出"，而非手掌朝向。
        #   作用：作为朝向锁的输入，区分手臂"朝前伸出"vs"侧向摆动"。
        #   注意：FDV 与真实手掌法向量在解剖学上通常垂直，不可混为一谈。
        def _forearm_proxy(side: str) -> np.ndarray:
            if side == "left":
                e_idx, w_idx = 7, 9
            else:
                e_idx, w_idx = 8, 10
            if (
                keypoints[e_idx, 2] < 0.3
                or keypoints[w_idx, 2] < 0.3
            ):
                return np.zeros(3, dtype=np.float32)
            forearm = keypoints[w_idx, :2] - keypoints[e_idx, :2]
            fn = float(np.linalg.norm(forearm)) + 1e-8
            return np.array(
                [forearm[0] / fn, forearm[1] / fn, 0.5],
                dtype=np.float32,
            )

        for side in ["right", "left"]:
            if side == "right":
                wl = right_wrist_local
                wl_other = left_wrist_local
                v_mag = right_velocity_mag
                theta1 = right_theta1
                theta2 = right_theta2
                ext_ratio = right_ext_ratio
                tnlf_valid = right_tnlf_valid
            else:
                wl = left_wrist_local
                wl_other = right_wrist_local
                v_mag = left_velocity_mag
                theta1 = left_theta1
                theta2 = left_theta2
                ext_ratio = left_ext_ratio
                tnlf_valid = left_tnlf_valid

            palm_normal = _forearm_proxy(side)

            gesture, confidence = self.engine.process_frame(
                track_id=track_id,
                side=side,
                wrist_local=wl if wl is not None else np.zeros(2, dtype=np.float32),
                velocity_mag=v_mag,
                theta1=theta1,
                theta2=theta2,
                ext_ratio=ext_ratio,
                palm_normal=palm_normal,
                tnlf_valid=tnlf_valid,
                timestamp=now,
                wrist_local_other=wl_other,
            )

            result = GestureResult(
                gesture_type=GestureType.WAVING if gesture == "waving" else GestureType.NONE,
                confidence=confidence,
            )

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result if best_result else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0
        )

    def reset(self) -> None:
        self.engine.reset_all()
        logger.info("TransformerGestureRecognizer 已重置")


# =============================================================================
# 简化手势引擎：鼻子+眼睛可见 + 手腕高于手肘 + TNLF手腕轨迹周期变化
# =============================================================================

class SimpleGestureEngine:
    """
    简化手势引擎（重做版）：

    1. 鼻子 + 至少一只眼睛可见
    2. 手腕高于手肘（图像坐标 wrist_y < elbow_y）
    3. TNLF 移动坐标系下手腕轨迹相对向量的正负性周期变化
       - 3帧滑动平均做抖动过滤
       - 对 wrist_local 主分量做 zero-crossing 分析
    """

    def __init__(
        self,
        nose_conf_threshold: float = 0.3,
        eye_conf_threshold: float = 0.3,
        wrist_elbow_conf_threshold: float = 0.3,
        period_window_seconds: float = 2.5,
        fps: float = 15.0,
        min_freq_hz: float = 0.35,
        max_freq_hz: float = 5.0,
        min_cycles: int = 2,
        min_confirm_frames: int = 3,
        hold_frames: int = 15,
        ema_alpha: float = 0.35,
        min_amplitude_tu: float = 0.03,
    ) -> None:
        self.nose_conf_threshold = nose_conf_threshold
        self.eye_conf_threshold = eye_conf_threshold
        self.wrist_elbow_conf_threshold = wrist_elbow_conf_threshold
        self.period_window = period_window_seconds
        self.fps = fps
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_cycles = min_cycles
        self.min_confirm_frames = min_confirm_frames
        self.hold_frames = hold_frames
        self.ema_alpha = ema_alpha
        self.min_amplitude_tu = min_amplitude_tu

        # TNLF wrist_local 历史（经滑动平均平滑后）
        self._wrist_history: Dict[str, deque] = {}
        # 5帧滑动平均窗口（原始 wrist_local）—— 过滤车辆高频颠簸
        self._smooth_win: Dict[str, deque] = {}
        self._confirm_count: Dict[str, int] = {}
        self._hold_count: Dict[str, int] = {}
        self._ema_conf: Dict[str, float] = {}
        self._last_result: Dict[str, Tuple[str, float]] = {}

    @staticmethod
    def _key(track_id: str, side: str) -> str:
        return f"{track_id}_{side}"

    def process_frame(
        self,
        keypoints: np.ndarray,
        side: str,
        track_id: str,
        wrist_local: Optional[Tuple[float, float]],
        timestamp: float,
    ) -> Tuple[str, float]:
        """处理单帧，返回 (gesture_type, confidence)。"""
        k = self._key(track_id, side)
        maxlen = int(self.period_window * self.fps)

        # ---- 1. 鼻子 + 至少一只眼睛可见 ----
        if keypoints.shape[0] < 17:
            self._reset(k)
            return "none", 0.0
        nose_conf = float(keypoints[0, 2])
        if nose_conf < self.nose_conf_threshold:
            self._reset(k)
            return "none", 0.0

        left_eye_conf = float(keypoints[1, 2]) if len(keypoints) > 1 else 0.0
        right_eye_conf = float(keypoints[2, 2]) if len(keypoints) > 2 else 0.0
        if left_eye_conf < self.eye_conf_threshold and right_eye_conf < self.eye_conf_threshold:
            self._reset(k)
            return "none", 0.0

        # ---- 2. 手腕高于手肘（图像坐标 y 越小越靠上） ----
        if side == "left":
            elbow_idx, wrist_idx = 7, 9
        else:
            elbow_idx, wrist_idx = 8, 10

        elbow_conf = float(keypoints[elbow_idx, 2])
        wrist_conf = float(keypoints[wrist_idx, 2])
        if elbow_conf < self.wrist_elbow_conf_threshold or wrist_conf < self.wrist_elbow_conf_threshold:
            self._reset(k)
            return "none", 0.0

        wrist_y = float(keypoints[wrist_idx, 1])
        elbow_y = float(keypoints[elbow_idx, 1])
        if wrist_y >= elbow_y:
            self._reset(k)
            return "none", 0.0

        # ---- 3. 手腕相对手肘向量周期运动 + 抖动过滤 ----
        # 使用 wrist - elbow（归一化到肩宽）代替 TNLF wrist_local。
        # 原因：wrist 与 elbow 是相邻关键点，YOLO 检测噪声高度相关，
        # 差分后噪声远小于 TNLF（wrist - origin），车辆整体平移时几乎为 0。
        shoulder_l = keypoints[5][:2]
        shoulder_r = keypoints[6][:2]
        shoulder_width = float(np.linalg.norm(shoulder_r - shoulder_l))
        if shoulder_width < 1.0:
            self._reset(k)
            return "none", 0.0

        wrist_elbow_vec = (keypoints[wrist_idx][:2] - keypoints[elbow_idx][:2]) / shoulder_width
        rel_vec = (float(wrist_elbow_vec[0]), float(wrist_elbow_vec[1]))

        # 5帧滑动平均去抖（过滤车辆悬挂高频颠簸）
        if k not in self._smooth_win:
            self._smooth_win[k] = deque(maxlen=5)
        self._smooth_win[k].append(rel_vec)
        smoothed = (
            sum(p[0] for p in self._smooth_win[k]) / len(self._smooth_win[k]),
            sum(p[1] for p in self._smooth_win[k]) / len(self._smooth_win[k]),
        )

        if k not in self._wrist_history:
            self._wrist_history[k] = deque(maxlen=maxlen)
        self._wrist_history[k].append(smoothed)

        is_periodic, raw_conf = self._detect_periodic(k)

        # 连续确认 / 衰减
        if is_periodic:
            self._confirm_count[k] = self._confirm_count.get(k, 0) + 1
        else:
            self._confirm_count[k] = max(0, self._confirm_count.get(k, 0) - 1)

        # EMA 平滑
        if k not in self._ema_conf:
            self._ema_conf[k] = raw_conf
        else:
            self._ema_conf[k] = (
                self.ema_alpha * raw_conf
                + (1.0 - self.ema_alpha) * self._ema_conf[k]
            )
        smoothed_conf = self._ema_conf[k]

        if self._confirm_count.get(k, 0) >= self.min_confirm_frames:
            self._hold_count[k] = self.hold_frames
            gesture, conf = "waving", smoothed_conf
            self._last_result[k] = (gesture, conf)
            return gesture, conf

        if self._hold_count.get(k, 0) > 0:
            self._hold_count[k] -= 1
            if k in self._last_result:
                decay = self._hold_count[k] / max(self.hold_frames, 1)
                return self._last_result[k][0], self._last_result[k][1] * decay

        return "none", smoothed_conf

    def _detect_periodic(self, k: str) -> Tuple[bool, float]:
        """检测 wrist_local 轨迹的周期性（基于 TNLF，去趋势 + zero-crossing）。"""
        history = self._wrist_history.get(k)
        min_hist = max(8, self.min_cycles * 4)
        if history is None or len(history) < min_hist:
            return False, 0.0

        data = np.array(list(history), dtype=np.float32)
        # 取方差更大的轴作为主分量
        x_var = float(np.var(data[:, 0]))
        y_var = float(np.var(data[:, 1]))
        axis = data[:, 1] if y_var > x_var else data[:, 0]

        # 去趋势（线性回归减趋势）
        x = np.arange(len(axis))
        slope, intercept = np.polyfit(x, axis, 1)
        detrended = axis - (slope * x + intercept)

        # Zero-crossing 计数
        signs = np.sign(detrended)
        zero_crossings = 0
        for i in range(1, len(signs)):
            if signs[i - 1] != 0 and signs[i] != 0 and signs[i - 1] * signs[i] < 0:
                zero_crossings += 1

        cycles = zero_crossings // 2

        # 频率估算
        duration = len(detrended) / max(self.fps, 1.0)
        freq = cycles / max(duration, 0.001)

        # 振幅（torso_units）
        amplitude = float((np.max(detrended) - np.min(detrended)) / 2.0)

        # 周期一致性（提前计算用于日志）
        crossing_indices = []
        for i in range(1, len(signs)):
            if signs[i - 1] != 0 and signs[i] != 0 and signs[i - 1] * signs[i] < 0:
                crossing_indices.append(i)

        consistency = 0.0
        if len(crossing_indices) >= 4:
            intervals = np.diff(crossing_indices).astype(np.float64)
            mean_int = float(np.mean(intervals))
            std_int = float(np.std(intervals))
            if mean_int >= 1.0:
                consistency = max(0.0, 1.0 - std_int / mean_int)

        # 逐步检查并记录失败原因
        reason = None
        if cycles < self.min_cycles:
            reason = f"cycles={cycles}<{self.min_cycles}"
        elif not (self.min_freq_hz <= freq <= self.max_freq_hz):
            reason = f"freq={freq:.2f}Hz out of range [{self.min_freq_hz},{self.max_freq_hz}]"
        elif amplitude < self.min_amplitude_tu:
            reason = f"amp={amplitude:.4f}tu < {self.min_amplitude_tu}"
        elif len(crossing_indices) < 4:
            reason = f"crossings={len(crossing_indices)}<4"
        elif consistency < 0.35:
            reason = f"consistency={consistency:.2f}<0.35"

        if reason:
            logger.debug(
                "simple-periodic[%s] REJECTED: %s | hist=%d zc=%d cycles=%d freq=%.2f amp=%.4f consist=%.2f",
                k, reason, len(history), zero_crossings, cycles, freq, amplitude, consistency,
            )
            return False, 0.0

        amp_score = min(1.0, amplitude / 0.3)
        conf = 0.4 + 0.3 * amp_score + 0.3 * consistency
        logger.debug(
            "simple-periodic[%s] ACCEPTED: conf=%.2f hist=%d zc=%d cycles=%d freq=%.2f amp=%.4f consist=%.2f",
            k, conf, len(history), zero_crossings, cycles, freq, amplitude, consistency,
        )
        return True, min(1.0, conf)

    def _reset(self, k: str) -> None:
        """重置单侧状态（含 hold_count / last_result 修复）。"""
        self._wrist_history.pop(k, None)
        self._smooth_win.pop(k, None)
        self._confirm_count.pop(k, None)
        self._hold_count.pop(k, None)
        self._ema_conf.pop(k, None)
        self._last_result.pop(k, None)

    def cleanup_stale(self, active_keys: set, max_age_seconds: float = 10.0) -> None:
        """清理不再活跃的 track/side 状态。"""
        for k in list(self._wrist_history.keys()):
            if k not in active_keys:
                self._reset(k)

    def reset_all(self) -> None:
        self._wrist_history.clear()
        self._smooth_win.clear()
        self._confirm_count.clear()
        self._hold_count.clear()
        self._ema_conf.clear()
        self._last_result.clear()

    def check_trigger_only(self, keypoints: np.ndarray) -> bool:
        """
        仅检查 Simple 引擎的前 3 个条件（面部 + 手肘高于手腕），
        不做周期性检测。用作 MiniCPM 引擎的 trigger。
        """
        if keypoints is None or len(keypoints) < 17:
            return False
        # 鼻子可见
        nose_conf = float(keypoints[0, 2])
        if nose_conf < self.nose_conf_threshold:
            logger.info("check_trigger: nose_conf=%.2f < %.2f", nose_conf, self.nose_conf_threshold)
            return False
        # 至少一只眼睛可见
        left_eye_conf = float(keypoints[1, 2]) if len(keypoints) > 1 else 0.0
        right_eye_conf = float(keypoints[2, 2]) if len(keypoints) > 2 else 0.0
        if left_eye_conf < self.eye_conf_threshold and right_eye_conf < self.eye_conf_threshold:
            logger.info("check_trigger: eye_conf left=%.2f right=%.2f < %.2f", left_eye_conf, right_eye_conf, self.eye_conf_threshold)
            return False
        # 至少一侧手腕高于手肘（允许 10px 容差，应对手臂弯曲时的投影误差）
        WRIST_ELBOW_CONF = 0.10
        WRIST_ELBOW_PIX_TOL = 10.0
        for side in ("left", "right"):
            elbow_idx, wrist_idx = (7, 9) if side == "left" else (8, 10)
            elbow_conf = float(keypoints[elbow_idx, 2])
            wrist_conf = float(keypoints[wrist_idx, 2])
            if elbow_conf < WRIST_ELBOW_CONF or wrist_conf < WRIST_ELBOW_CONF:
                logger.info("check_trigger %s: conf elbow=%.2f wrist=%.2f < %.2f", side, elbow_conf, wrist_conf, WRIST_ELBOW_CONF)
                continue
            wrist_y = float(keypoints[wrist_idx, 1])
            elbow_y = float(keypoints[elbow_idx, 1])
            if wrist_y <= elbow_y + WRIST_ELBOW_PIX_TOL:
                logger.info("check_trigger %s: triggered wrist_y=%.1f <= elbow_y=%.1f + %.1f", side, wrist_y, elbow_y, WRIST_ELBOW_PIX_TOL)
                return True
            else:
                logger.info("check_trigger %s: wrist_y=%.1f > elbow_y=%.1f + %.1f, not triggered", side, wrist_y, elbow_y, WRIST_ELBOW_PIX_TOL)
        return False


class SimpleGestureRecognizer:
    """
    简化手势识别器包装，供 detector.py 调用。
    """

    def __init__(self, nose_conf_threshold: float = 0.25, eye_conf_threshold: float = 0.25,
                 wrist_elbow_conf_threshold: float = 0.3,
                 period_window_seconds: float = 2.5,
                 fps: float = 15.0, min_freq_hz: float = 0.35, max_freq_hz: float = 3.0,
                 min_cycles: int = 2, min_confirm_frames: int = 3, hold_frames: int = 15,
                 ema_alpha: float = 0.35, min_amplitude_tu: float = 0.05) -> None:
        self.engine = SimpleGestureEngine(
            nose_conf_threshold=nose_conf_threshold,
            eye_conf_threshold=eye_conf_threshold,
            wrist_elbow_conf_threshold=wrist_elbow_conf_threshold,
            period_window_seconds=period_window_seconds,
            fps=fps,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            min_cycles=min_cycles,
            min_confirm_frames=min_confirm_frames,
            hold_frames=hold_frames,
            ema_alpha=ema_alpha,
            min_amplitude_tu=min_amplitude_tu,
        )
        logger.info(
            "SimpleGestureRecognizer: nose>%.2f eye>%.2f wrist>elbow + TNLF periodic",
            self.engine.nose_conf_threshold,
            self.engine.eye_conf_threshold,
        )

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        **_kwargs,
    ) -> "GestureResult":
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 清理过期状态
        if active_track_ids:
            active_keys = {
                f"{tid}_{side}"
                for tid in active_track_ids
                for side in ["left", "right"]
            }
            self.engine.cleanup_stale(active_keys)

        best: Optional[GestureResult] = None
        for side in ["right", "left"]:
            wl = left_wrist_local if side == "left" else right_wrist_local
            wl_tuple = (float(wl[0]), float(wl[1])) if wl is not None else None

            gesture, conf = self.engine.process_frame(
                keypoints=keypoints,
                side=side,
                track_id=track_id,
                wrist_local=wl_tuple,
                timestamp=now,
            )
            r = GestureResult(
                gesture_type=GestureType.WAVING if gesture == "waving" else GestureType.NONE,
                confidence=conf,
            )
            if best is None or r.confidence > best.confidence:
                best = r

        return best if best else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0,
        )

    def reset(self) -> None:
        self.engine.reset_all()

def _choose_recognizer() -> "GestureRecognizer | TransformerGestureRecognizer | SimpleGestureRecognizer | SimpleTransformerHybridRecognizer | HybridGestureRecognizer | SimpleMiniCPMHybridRecognizer":
    """根据配置选择手势识别引擎。"""
    config = get_config()
    engine_mode = config.ai.gesture_engine.lower()

    if engine_mode == "simple":
        logger.info("使用 Simple 手势引擎（3条件规则）")
        return SimpleGestureRecognizer(
            nose_conf_threshold=config.ai.gesture_facing_hard_threshold,
            period_window_seconds=2.5,
            fps=config.stream.fps,
            min_freq_hz=config.ai.gesture_period_min_freq,
            max_freq_hz=config.ai.gesture_period_max_freq,
            min_cycles=config.ai.gesture_period_min_cycles,
            min_confirm_frames=config.ai.gesture_pose_min_frames,
            hold_frames=config.ai.gesture_hold_max_frames,
            ema_alpha=config.ai.gesture_ema_alpha,
        )
    elif engine_mode == "simple-transformer":
        logger.info("使用 Simple+Transformer 混合手势引擎")
        try:
            return SimpleTransformerHybridRecognizer(
                transformer_model_path=config.ai.transformer_model_path,
                transformer_threshold=config.ai.transformer_confidence_threshold,
            )
        except Exception as e:
            logger.warning("Simple+Transformer 混合引擎初始化失败 (%s)，回退到 Simple 引擎", e)
            return SimpleGestureRecognizer()
    elif engine_mode == "transformer":
        try:
            return TransformerGestureRecognizer(
                model_path=config.ai.transformer_model_path,
                confidence_threshold=config.ai.transformer_confidence_threshold,
                ema_alpha=config.ai.gesture_ema_alpha,
                hold_frames=config.ai.gesture_hold_max_frames,
            )
        except Exception as e:
            logger.warning(
                "Transformer 引擎初始化失败 (%s)，回退到 TripleLock 引擎", e
            )
            return GestureRecognizer()
    elif engine_mode == "hybrid":
        # Hybrid 模式：transformer + triplelock 双重验证
        try:
            return HybridGestureRecognizer(
                transformer_model_path=config.ai.transformer_model_path,
                transformer_threshold=config.ai.transformer_confidence_threshold,
            )
        except Exception as e:
            logger.warning("Hybrid 引擎初始化失败 (%s)，回退到 TripleLock 引擎", e)
            return GestureRecognizer()
    elif engine_mode == "simple-minicpm":
        logger.info("使用 Simple+MiniCPM 混合手势引擎")
        try:
            return SimpleMiniCPMHybridRecognizer(
                model_path=config.ai.minicpm_model_path,
                input_size=config.ai.minicpm_input_size,
                max_frames=config.ai.minicpm_max_frames,
                buffer_seconds=config.ai.minicpm_buffer_seconds,
                min_frames=config.ai.minicpm_min_frames,
                infer_interval_s=config.ai.minicpm_infer_interval_s,
                max_concurrent=config.ai.minicpm_max_concurrent,
                confidence_threshold=config.ai.minicpm_confidence_threshold,
                cooldown_seconds=config.ai.minicpm_cooldown_seconds,
                prompt=config.ai.minicpm_prompt,
            )
        except Exception as e:
            logger.warning("Simple+MiniCPM 混合引擎初始化失败 (%s)，回退到 Simple 引擎", e)
            return SimpleGestureRecognizer()
    elif engine_mode == "minicpm":
        logger.info("使用纯 MiniCPM 手势引擎")
        try:
            return MiniCPMGestureRecognizer(
                model_path=config.ai.minicpm_model_path,
                input_size=config.ai.minicpm_input_size,
                max_frames=config.ai.minicpm_max_frames,
                buffer_seconds=config.ai.minicpm_buffer_seconds,
                min_frames=config.ai.minicpm_min_frames,
                infer_interval_s=config.ai.minicpm_infer_interval_s,
                max_concurrent=config.ai.minicpm_max_concurrent,
                confidence_threshold=config.ai.minicpm_confidence_threshold,
                cooldown_seconds=config.ai.minicpm_cooldown_seconds,
                prompt=config.ai.minicpm_prompt,
            )
        except Exception as e:
            logger.warning("MiniCPM 引擎初始化失败 (%s)，回退到 Simple 引擎", e)
            return SimpleGestureRecognizer()
    else:
        return GestureRecognizer()


class HybridGestureRecognizer:
    """
    混合手势识别器：Transformer + TripleLock 双重验证。

    - Transformer 作为主识别器，提供高召回率
    - TripleLock 作为验证门，过滤假阳性
    - 仅当两者同时确认 "waving" 时才输出 positive
    """

    def __init__(self, transformer_model_path: str,
                 transformer_threshold: float = 0.5) -> None:
        self.transformer = TransformerGestureRecognizer(
            model_path=transformer_model_path,
            confidence_threshold=transformer_threshold,
        )
        self.triplelock = GestureRecognizer()
        logger.info(
            "HybridGestureRecognizer: transformer_threshold=%.2f + triplelock verification",
            transformer_threshold,
        )

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        left_tnlf_valid: bool = False,
        right_tnlf_valid: bool = False,
        left_velocity_mag: float = 0.0,
        right_velocity_mag: float = 0.0,
        left_theta1: float = 0.0, left_theta2: float = 0.0, left_ext_ratio: float = 0.0,
        right_theta1: float = 0.0, right_theta2: float = 0.0, right_ext_ratio: float = 0.0,
    ) -> "GestureResult":
        # 始终调用 Transformer，保持其 45 帧滑窗连续积累；否则 TripleLock 缄默时
        # 模型饥饿，永远输出 NONE。
        tf_result = self.transformer.recognize(
            keypoints, track_id, left_palm_normal, right_palm_normal,
            frame_timestamp, active_track_ids,
            left_wrist_local=left_wrist_local,
            right_wrist_local=right_wrist_local,
            left_tnlf_valid=left_tnlf_valid,
            right_tnlf_valid=right_tnlf_valid,
            left_velocity_mag=left_velocity_mag,
            right_velocity_mag=right_velocity_mag,
            left_theta1=left_theta1, left_theta2=left_theta2, left_ext_ratio=left_ext_ratio,
            right_theta1=right_theta1, right_theta2=right_theta2, right_ext_ratio=right_ext_ratio,
        )

        tl_result = self.triplelock.recognize(
            keypoints, track_id, left_palm_normal, right_palm_normal,
            frame_timestamp, active_track_ids,
        )

        # Hybrid: both must agree
        if tl_result.gesture_type == GestureType.WAVING and tf_result.gesture_type == GestureType.WAVING:
            avg_conf = (tl_result.confidence + tf_result.confidence) / 2.0
            return GestureResult(gesture_type=GestureType.WAVING, confidence=avg_conf)

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self) -> None:
        self.transformer.reset()
        self.triplelock.reset()


class SimpleTransformerHybridRecognizer:
    """
    Simple + Transformer 混合手势识别器。

    - Simple (3条件规则) 作为快速预筛选
    - Transformer 作为二次验证
    - 仅当两者同时确认 "waving" 时才输出 positive
    """

    def __init__(self, transformer_model_path: str,
                 transformer_threshold: float = 0.5) -> None:
        self.simple = SimpleGestureRecognizer()
        self._nose_threshold = get_config().ai.gesture_facing_hard_threshold
        try:
            self.transformer = TransformerGestureRecognizer(
                model_path=transformer_model_path,
                confidence_threshold=transformer_threshold,
            )
            self._has_transformer = True
        except Exception as e:
            logger.warning("Transformer 加载失败 (%s)，回退到纯 Simple 模式", e)
            self._has_transformer = False
        logger.info(
            "SimpleTransformerHybridRecognizer: simple + transformer threshold=%.2f",
            transformer_threshold,
        )

    def _check_hard_pose_rules(self, keypoints: np.ndarray) -> bool:
        """最硬性规则：鼻子+眼睛可见，且至少一侧手腕高于手肘。"""
        if keypoints is None or len(keypoints) < 17:
            return False

        # 鼻子可见
        if float(keypoints[0, 2]) < self._nose_threshold:
            return False

        # 至少一只眼睛可见
        left_eye_conf = float(keypoints[1, 2]) if len(keypoints) > 1 else 0.0
        right_eye_conf = float(keypoints[2, 2]) if len(keypoints) > 2 else 0.0
        if left_eye_conf < self._nose_threshold and right_eye_conf < self._nose_threshold:
            return False

        # 至少一侧手腕高于手肘（且关键点置信度足够）
        for side in ("left", "right"):
            elbow_idx, wrist_idx = (7, 9) if side == "left" else (8, 10)
            if float(keypoints[elbow_idx, 2]) < 0.3 or float(keypoints[wrist_idx, 2]) < 0.3:
                continue
            if float(keypoints[wrist_idx, 1]) < float(keypoints[elbow_idx, 1]):
                return True

        return False

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        left_tnlf_valid: bool = False,
        right_tnlf_valid: bool = False,
        left_velocity_mag: float = 0.0,
        right_velocity_mag: float = 0.0,
        left_theta1: float = 0.0, left_theta2: float = 0.0, left_ext_ratio: float = 0.0,
        right_theta1: float = 0.0, right_theta2: float = 0.0, right_ext_ratio: float = 0.0,
    ) -> "GestureResult":
        # 始终调用 Transformer，保持其 45 帧滑窗连续积累；否则仅在 Simple
        # 已经 WAVING 的瞬间才喂帧，window 永远填不满，模型恒回 NONE。
        if self._has_transformer:
            tf_result = self.transformer.recognize(
                keypoints, track_id, left_palm_normal, right_palm_normal,
                frame_timestamp, active_track_ids,
                left_wrist_local=left_wrist_local,
                right_wrist_local=right_wrist_local,
                left_tnlf_valid=left_tnlf_valid,
                right_tnlf_valid=right_tnlf_valid,
                left_velocity_mag=left_velocity_mag,
                right_velocity_mag=right_velocity_mag,
                left_theta1=left_theta1, left_theta2=left_theta2, left_ext_ratio=left_ext_ratio,
                right_theta1=right_theta1, right_theta2=right_theta2, right_ext_ratio=right_ext_ratio,
            )
        else:
            tf_result = None

        s_result = self.simple.recognize(
            keypoints, track_id, left_palm_normal, right_palm_normal,
            frame_timestamp, active_track_ids,
            left_wrist_local=left_wrist_local,
            right_wrist_local=right_wrist_local,
        )

        if tf_result is None:
            return s_result

        # Simple 为主检测器：Simple 确认后才进入 Transformer 辅助验证
        if s_result.gesture_type != GestureType.WAVING:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        # Simple 已确认 waving
        if tf_result.gesture_type == GestureType.WAVING:
            # 两者一致，取更高置信度
            return GestureResult(
                gesture_type=GestureType.WAVING,
                confidence=max(s_result.confidence, tf_result.confidence),
            )

        # Simple 确认但 Transformer 未确认：
        # Transformer 对低幅度招手不敏感，此处保留 Simple 结果但降权
        if s_result.confidence > 0.45:
            return GestureResult(
                gesture_type=GestureType.WAVING,
                confidence=s_result.confidence * 0.9,
            )

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self) -> None:
        self.simple.reset()
        if self._has_transformer:
            self.transformer.reset()


class SimpleMiniCPMHybridRecognizer:
    """
    Simple + MiniCPM 混合手势识别器（流式视频推理）。

    - Simple（面部 + 手肘高于手腕）做 trigger
    - trigger 满足后持续积累人物区域的视频帧（滑动窗口）
    - 定期将窗口帧送入 MiniCPM 做视频级招手判定
    - 结果异步更新，recognize() 立即返回缓存结果，不阻塞主流程
    """

    def __init__(
        self,
        model_path: str = "OpenBMB/MiniCPM-V-4.6",
        input_size: int = 448,
        max_frames: int = 16,
        buffer_seconds: float = 1.5,
        min_frames: int = 8,
        infer_interval_s: float = 1.0,
        max_concurrent: int = 16,
        confidence_threshold: float = 0.6,
        cooldown_seconds: float = 2.0,
        prompt: Optional[str] = None,
        minicpm_engine: Optional[Any] = None,
    ) -> None:
        # Simple trigger 阈值放得很宽：只要能看到鼻子/眼睛 + 手腕在手肘上方即可触发
        self._simple = SimpleGestureEngine(
            nose_conf_threshold=0.15,
            eye_conf_threshold=0.15,
        )
        self._input_size = input_size
        self._max_frames = max_frames
        self._buffer_seconds = buffer_seconds
        self._min_frames = min_frames
        self._infer_interval_s = infer_interval_s
        self._cooldown_seconds = cooldown_seconds

        # 帧缓冲区：track_id -> deque of (timestamp, cropped_frame, frame_idx)
        self._buffers: Dict[str, deque] = {}
        self._last_infer_ts: Dict[str, float] = {}
        self._cooldown_until: Dict[str, float] = {}
        # 推理窗口记录：track_id -> List[(start_frame_idx, end_frame_idx)]
        self._inference_windows: Dict[str, List[Tuple[int, int]]] = {}

        # MiniCPM 推理引擎（支持外部传入共享实例，避免多实例并发加载导致模型损坏）
        if minicpm_engine is not None:
            self._minicpm = minicpm_engine
            self._has_minicpm = True
            logger.info(
                "SimpleMiniCPMHybridRecognizer: simple trigger + 共享 MiniCPM "
                "input=%d max_frames=%d buffer=%.1fs min_frames=%d interval=%.2fs",
                input_size, max_frames, buffer_seconds, min_frames, infer_interval_s,
            )
        else:
            try:
                from app.ai.minicpm_engine import MiniCPMEngine
                self._minicpm = MiniCPMEngine(
                    model_path=model_path,
                    max_concurrent=max_concurrent,
                    confidence_threshold=confidence_threshold,
                    prompt=prompt,
                )
                self._has_minicpm = True
                logger.info(
                    "SimpleMiniCPMHybridRecognizer: simple trigger + MiniCPM(%s) "
                    "input=%d max_frames=%d buffer=%.1fs min_frames=%d interval=%.2fs concurrent=%d",
                    model_path, input_size, max_frames, buffer_seconds, min_frames,
                    infer_interval_s, max_concurrent,
                )
            except Exception as e:
                logger.warning("MiniCPM 引擎初始化失败 (%s)，回退到纯 Simple 模式", e)
                self._minicpm = None
                self._has_minicpm = False

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        left_tnlf_valid: bool = False,
        right_tnlf_valid: bool = False,
        left_velocity_mag: float = 0.0,
        right_velocity_mag: float = 0.0,
        left_theta1: float = 0.0, left_theta2: float = 0.0, left_ext_ratio: float = 0.0,
        right_theta1: float = 0.0, right_theta2: float = 0.0, right_ext_ratio: float = 0.0,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: int = 0,
        **_kwargs,
    ) -> "GestureResult":
        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 1. Simple trigger：面部 + 手肘高于手腕
        triggered = self._simple.check_trigger_only(keypoints)

        # 2. 清理过期状态
        if active_track_ids is not None:
            stale = [
                tid for tid in list(self._buffers.keys())
                if tid not in active_track_ids
            ]
            for tid in stale:
                self._buffers.pop(tid, None)
                self._last_infer_ts.pop(tid, None)
                self._cooldown_until.pop(tid, None)
                self._inference_windows.pop(tid, None)
            if self._has_minicpm:
                self._minicpm.cleanup_stale(active_track_ids)

        if not triggered:
            # trigger 不满足，清空该 track 的缓冲区
            if track_id in self._buffers:
                del self._buffers[track_id]
                self._last_infer_ts.pop(track_id, None)
                self._inference_windows.pop(track_id, None)
            logger.debug("SimpleMiniCPM[%s]: trigger=false, 直接返回 NONE", track_id)
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        logger.debug("SimpleMiniCPM[%s]: trigger=true, 积累帧", track_id)

        # 3. 积累帧
        if frame is not None and bbox is not None:
            self._buffer_frame(track_id, frame, bbox, now, frame_idx)

        # 4. 检查是否满足推理条件
        self._maybe_submit(track_id, now)

        # 5. 返回 MiniCPM 缓存结果（或 CHECKING）
        if self._has_minicpm:
            gesture, conf = self._minicpm.get_result_at(track_id, frame_idx)
            logger.info(
                "SimpleMiniCPM[%s]: get_result_at frame_idx=%d gesture=%s conf=%.2f",
                track_id, frame_idx, gesture, conf,
            )
            if gesture == "waving":
                return GestureResult(gesture_type=GestureType.WAVING, confidence=conf)
            # trigger 已满足且正在积累帧 / 等待推理结果：显示"检测中"
            if track_id in self._buffers:
                return GestureResult(gesture_type=GestureType.CHECKING, confidence=0.5)
        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def _buffer_frame(
        self,
        track_id: str,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        frame_idx: int = 0,
    ) -> None:
        """将人物区域裁剪并缩放后存入滑动窗口。"""
        x1, y1, x2, y2 = bbox
        # 边界检查
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # 等比例缩放到短边为 input_size
        ch, cw = crop.shape[:2]
        scale = self._input_size / min(ch, cw)
        new_h, new_w = int(ch * scale), int(cw * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 初始化缓冲区
        if track_id not in self._buffers:
            max_len = int(self._buffer_seconds * 15) + 5  # 按 15fps 估算，留余量
            self._buffers[track_id] = deque(maxlen=max(max_len, 30))

        self._buffers[track_id].append((timestamp, resized, frame_idx))

    def _maybe_submit(self, track_id: str, now: float) -> None:
        """检查是否满足 MiniCPM 推理条件，满足则提交。"""
        if not self._has_minicpm:
            logger.info("SimpleMiniCPM[%s]: _maybe_submit skip, _has_minicpm=False", track_id)
            return

        buf = self._buffers.get(track_id)
        if buf is None:
            logger.info("SimpleMiniCPM[%s]: _maybe_submit skip, buf=None", track_id)
            return
        if len(buf) < self._min_frames:
            logger.info("SimpleMiniCPM[%s]: _maybe_submit skip, buf_len=%d < min_frames=%d", track_id, len(buf), self._min_frames)
            return

        # 冷却期检查
        cooldown = self._cooldown_until.get(track_id, 0)
        if now < cooldown:
            logger.info("SimpleMiniCPM[%s]: _maybe_submit skip, in cooldown %.1fs", track_id, cooldown - now)
            return

        # 推理间隔检查
        last_ts = self._last_infer_ts.get(track_id, 0)
        if now - last_ts < self._infer_interval_s:
            logger.info("SimpleMiniCPM[%s]: _maybe_submit skip, interval %.2fs < %.2fs", track_id, now - last_ts, self._infer_interval_s)
            return

        # 均匀抽帧
        frames = [f for _ts, f, _idx in buf]
        frame_indices = [idx for _ts, _f, idx in buf]
        if len(frames) > self._max_frames:
            indices = np.linspace(0, len(frames) - 1, self._max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            used_indices = [frame_indices[i] for i in indices]
        else:
            used_indices = frame_indices

        self._last_infer_ts[track_id] = now
        self._minicpm.submit(track_id, frames, frame_indices=used_indices)
        # 记录本次推理覆盖的帧窗口
        window_start = int(min(used_indices))
        window_end = int(max(used_indices))
        self._inference_windows.setdefault(track_id, []).append((window_start, window_end))
        logger.info(
            "SimpleMiniCPM[%s]: 提交推理, frames=%d, window=[%d, %d]",
            track_id, len(frames), window_start, window_end,
        )

        # 如果 MiniCPM 已确认 waving，设置冷却期
        gesture, conf = self._minicpm.get_result(track_id)
        if gesture == "waving" and conf >= 0.6:
            self._cooldown_until[track_id] = now + self._cooldown_seconds

    def get_inference_windows(self, track_id: str) -> List[Tuple[int, int, str, float]]:
        """获取该 track 的所有推理窗口及其结果。"""
        if self._has_minicpm and self._minicpm is not None:
            return self._minicpm.get_inference_windows(track_id)
        return []

    def wait_for_result(self, track_id: str, timeout: float = 30.0) -> Tuple[str, float]:
        """阻塞等待该 track 的 MiniCPM 推理完成。"""
        if not self._has_minicpm or self._minicpm is None:
            return "none", 0.0
        return self._minicpm.wait_for_result(track_id, timeout=timeout)

    def get_buffer_length(self, track_id: str) -> int:
        """获取该 track 的帧缓冲区长度。"""
        buf = self._buffers.get(track_id)
        return len(buf) if buf else 0

    def reset(self) -> None:
        self._simple.reset_all()
        self._buffers.clear()
        self._last_infer_ts.clear()
        self._cooldown_until.clear()
        self._inference_windows.clear()
        if self._has_minicpm and self._minicpm:
            self._minicpm.reset()


class MiniCPMGestureRecognizer:
    """
    纯 MiniCPM 手势识别器（无 Simple trigger，直接视频推理）。

    - 只要有 bbox 和 frame 就持续积累人物区域的视频帧（滑动窗口）
    - 定期将窗口帧送入 MiniCPM 做视频级招手判定
    - 结果异步更新，recognize() 立即返回缓存结果，不阻塞主流程
    """

    def __init__(
        self,
        model_path: str = "OpenBMB/MiniCPM-V-4.6",
        input_size: int = 448,
        max_frames: int = 16,
        buffer_seconds: float = 1.5,
        min_frames: int = 8,
        infer_interval_s: float = 1.0,
        max_concurrent: int = 16,
        confidence_threshold: float = 0.6,
        cooldown_seconds: float = 2.0,
        prompt: Optional[str] = None,
        minicpm_engine: Optional[Any] = None,
    ) -> None:
        self._input_size = input_size
        self._max_frames = max_frames
        self._buffer_seconds = buffer_seconds
        self._min_frames = min_frames
        self._infer_interval_s = infer_interval_s
        self._cooldown_seconds = cooldown_seconds

        # 帧缓冲区：track_id -> deque of (timestamp, cropped_frame, frame_idx)
        self._buffers: Dict[str, deque] = {}
        self._last_infer_ts: Dict[str, float] = {}
        self._cooldown_until: Dict[str, float] = {}
        # 推理窗口记录：track_id -> List[(start_frame_idx, end_frame_idx)]
        self._inference_windows: Dict[str, List[Tuple[int, int]]] = {}

        # MiniCPM 推理引擎（支持外部传入共享实例，避免多实例并发加载导致模型损坏）
        if minicpm_engine is not None:
            self._minicpm = minicpm_engine
            self._has_minicpm = True
            logger.info(
                "MiniCPMGestureRecognizer: 纯 共享 MiniCPM "
                "input=%d max_frames=%d buffer=%.1fs min_frames=%d interval=%.2fs",
                input_size, max_frames, buffer_seconds, min_frames, infer_interval_s,
            )
        else:
            try:
                from app.ai.minicpm_engine import MiniCPMEngine
                self._minicpm = MiniCPMEngine(
                    model_path=model_path,
                    max_concurrent=max_concurrent,
                    confidence_threshold=confidence_threshold,
                    prompt=prompt,
                )
                self._has_minicpm = True
                logger.info(
                    "MiniCPMGestureRecognizer: 纯 MiniCPM(%s) input=%d max_frames=%d "
                    "buffer=%.1fs min_frames=%d interval=%.2fs concurrent=%d",
                    model_path, input_size, max_frames, buffer_seconds, min_frames,
                    infer_interval_s, max_concurrent,
                )
            except Exception as e:
                logger.warning("MiniCPM 引擎初始化失败 (%s)，回退到 NONE 模式", e)
                self._minicpm = None
                self._has_minicpm = False

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
        left_wrist_local: Optional[np.ndarray] = None,
        right_wrist_local: Optional[np.ndarray] = None,
        left_tnlf_valid: bool = False,
        right_tnlf_valid: bool = False,
        left_velocity_mag: float = 0.0,
        right_velocity_mag: float = 0.0,
        left_theta1: float = 0.0, left_theta2: float = 0.0, left_ext_ratio: float = 0.0,
        right_theta1: float = 0.0, right_theta2: float = 0.0, right_ext_ratio: float = 0.0,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: int = 0,
        **_kwargs,
    ) -> "GestureResult":
        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 1. 清理过期状态
        if active_track_ids is not None:
            stale = [
                tid for tid in list(self._buffers.keys())
                if tid not in active_track_ids
            ]
            for tid in stale:
                self._buffers.pop(tid, None)
                self._last_infer_ts.pop(tid, None)
                self._cooldown_until.pop(tid, None)
                self._inference_windows.pop(tid, None)
            if self._has_minicpm:
                self._minicpm.cleanup_stale(active_track_ids)

        # 2. 无帧数据则无法推理
        if frame is None or bbox is None:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        # 3. 积累帧
        self._buffer_frame(track_id, frame, bbox, now, frame_idx)

        # 4. 检查是否满足推理条件
        self._maybe_submit(track_id, now)

        # 5. 返回 MiniCPM 缓存结果
        if self._has_minicpm:
            gesture, conf = self._minicpm.get_result_at(track_id, frame_idx)
            if gesture == "waving":
                return GestureResult(gesture_type=GestureType.WAVING, confidence=conf)
            # 正在积累帧 / 等待推理结果：显示"检测中"
            if track_id in self._buffers:
                return GestureResult(gesture_type=GestureType.CHECKING, confidence=0.5)
        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def _buffer_frame(
        self,
        track_id: str,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        frame_idx: int = 0,
    ) -> None:
        """将人物区域裁剪并缩放后存入滑动窗口。"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        ch, cw = crop.shape[:2]
        scale = self._input_size / min(ch, cw)
        new_h, new_w = int(ch * scale), int(cw * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if track_id not in self._buffers:
            max_len = int(self._buffer_seconds * 15) + 5
            self._buffers[track_id] = deque(maxlen=max(max_len, 30))

        self._buffers[track_id].append((timestamp, resized, frame_idx))

    def _maybe_submit(self, track_id: str, now: float) -> None:
        """检查是否满足 MiniCPM 推理条件，满足则提交。"""
        if not self._has_minicpm:
            return

        buf = self._buffers.get(track_id)
        if buf is None:
            return
        if len(buf) < self._min_frames:
            return

        cooldown = self._cooldown_until.get(track_id, 0)
        if now < cooldown:
            return

        last_ts = self._last_infer_ts.get(track_id, 0)
        if now - last_ts < self._infer_interval_s:
            return

        frames = [f for _ts, f, _idx in buf]
        frame_indices = [idx for _ts, _f, idx in buf]
        if len(frames) > self._max_frames:
            indices = np.linspace(0, len(frames) - 1, self._max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            used_indices = [frame_indices[i] for i in indices]
        else:
            used_indices = frame_indices

        self._last_infer_ts[track_id] = now
        self._minicpm.submit(track_id, frames, frame_indices=used_indices)
        window_start = int(min(used_indices))
        window_end = int(max(used_indices))
        self._inference_windows.setdefault(track_id, []).append((window_start, window_end))

        gesture, conf = self._minicpm.get_result(track_id)
        if gesture == "waving" and conf >= 0.6:
            self._cooldown_until[track_id] = now + self._cooldown_seconds

    def get_inference_windows(self, track_id: str) -> List[Tuple[int, int, str, float]]:
        """获取该 track 的所有推理窗口及其结果。"""
        if self._has_minicpm and self._minicpm is not None:
            return self._minicpm.get_inference_windows(track_id)
        return []

    def wait_for_result(self, track_id: str, timeout: float = 30.0) -> Tuple[str, float]:
        """阻塞等待该 track 的 MiniCPM 推理完成。"""
        if not self._has_minicpm or self._minicpm is None:
            return "none", 0.0
        return self._minicpm.wait_for_result(track_id, timeout=timeout)

    def get_buffer_length(self, track_id: str) -> int:
        """获取该 track 的帧缓冲区长度。"""
        buf = self._buffers.get(track_id)
        return len(buf) if buf else 0

    def reset(self) -> None:
        self._buffers.clear()
        self._last_infer_ts.clear()
        self._cooldown_until.clear()
        self._inference_windows.clear()
        if self._has_minicpm and self._minicpm:
            self._minicpm.reset()


# =============================================================================
# 手势类型与结果定义
# =============================================================================

class GestureType(str, Enum):
    NONE = "none"
    WAVING = "waving"
    GREETING = "greeting"
    HAILING = "hailing"
    HAND_UP = "hand_up"
    CHECKING = "checking"  # MiniCPM 正在分析中


@dataclass
class GestureResult:
    gesture_type: GestureType = GestureType.NONE
    confidence: float = 0.0
    wrist_pos: Optional[Tuple[float, float]] = None


# =============================================================================
# 全局单例 + 便捷函数
# =============================================================================

_recognizer: Optional[GestureRecognizer] = None


def get_recognizer() -> GestureRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = GestureRecognizer()
    return _recognizer


def is_hailing_gesture(
    keypoints: np.ndarray,
    track_id: str = "default",
    left_palm_normal: Optional[np.ndarray] = None,
    right_palm_normal: Optional[np.ndarray] = None,
    frame_timestamp: Optional[float] = None,
    active_track_ids: Optional[set] = None,
) -> Tuple[str, float]:
    """便捷函数。"""
    recognizer = get_recognizer()
    result = recognizer.recognize(
        keypoints, track_id,
        left_palm_normal, right_palm_normal,
        frame_timestamp,
        active_track_ids=active_track_ids,
    )
    return result.gesture_type.value, result.confidence
