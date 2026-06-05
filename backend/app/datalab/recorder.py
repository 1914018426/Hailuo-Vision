"""
素材录制器

在实时推流过程中，选择性录制原始帧、关键点序列、TNLF 特征和主引擎输出。
支持手动触发、自动手势触发和连续分段录制三种模式。

多摄像头支持：每个摄像头拥有独立的录制会话，帧数据不会混合。
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.datalab.models import RecordingSession, RecordingTriggerMode, ManualLabel
from app.datalab.persistence import DataLabStorage
from app.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RecorderStatus:
    """录制器实时状态。"""

    is_recording: bool = False
    session_id: Optional[str] = None
    camera_id: Optional[str] = None
    trigger_mode: Optional[str] = None
    frame_count: int = 0
    duration_s: float = 0.0
    person_count_peak: int = 0
    segment_count: int = 0
    # 多摄像头：所有活跃会话列表
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    total_active: int = 0


class GestureRecorder:
    """手势素材录制器（支持多摄像头并行录制）。"""

    # 连续录制默认分段长度：30 秒 @ 15fps = 450 帧
    SEGMENT_FRAMES_DEFAULT = 450

    def __init__(self, storage: DataLabStorage) -> None:
        self.storage = storage
        self.config = get_config()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 每个摄像头独立的会话状态（key = camera_id）
        self._sessions: Dict[str, RecordingSession] = {}
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._start_times: Dict[str, float] = {}
        self._frame_counts: Dict[str, int] = {}
        self._person_count_peaks: Dict[str, int] = {}

        # 连续录制状态（key = camera_id）
        self._segment_counts: Dict[str, int] = {}
        self._continuous_save_video: Dict[str, bool] = {}

        self._status = RecorderStatus()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(
        self,
        camera_id: str,
        trigger_mode: str = "manual",
        save_video: bool = True,
    ) -> RecordingSession:
        """开始录制指定摄像头。"""
        async with self._lock:
            if camera_id in self._sessions:
                raise RuntimeError(f"摄像头 {camera_id} 已有录制在进行中")

            # 保存事件循环引用，供非 async 线程提交分段任务
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

            session = self.storage.create_recording(
                camera_id=camera_id,
                trigger_mode=trigger_mode,
            )
            self._sessions[camera_id] = session
            self._start_times[camera_id] = time.time()
            self._frame_counts[camera_id] = 0
            self._person_count_peaks[camera_id] = 0
            self._segment_counts[camera_id] = 0
            self._continuous_save_video[camera_id] = save_video

            if save_video:
                video_path = str(
                    self.storage._recording_dir(session.id, session.start_time)
                    / "video.mp4"
                )
                session.video_path = video_path
                # VideoWriter 延迟初始化（等第一帧到来时知道尺寸）
                self._video_writers[camera_id] = None

            self.storage.update_recording(session)
            self._update_status()
            logger.info(
                "录制开始: id=%s camera=%s mode=%s",
                session.id,
                camera_id,
                trigger_mode,
            )
            return session

    async def stop(self, camera_id: Optional[str] = None) -> Optional[RecordingSession]:
        """
        停止录制。

        Args:
            camera_id: 要停止的摄像头。若为 None，则停止最早开始的一个活跃会话
                      （兼容旧版单录制行为）。
        """
        async with self._lock:
            if not self._sessions:
                return None

            if camera_id is None:
                # 默认停止第一个活跃会话（兼容旧 API）
                camera_id = next(iter(self._sessions))

            if camera_id not in self._sessions:
                return None

            session = self._sessions.pop(camera_id)
            session.end_time = time.time()
            session.duration_s = round(session.end_time - self._start_times.pop(camera_id, session.end_time), 2)
            session.frame_count = self._frame_counts.pop(camera_id, 0)
            session.person_count = self._person_count_peaks.pop(camera_id, 0)
            session.status = "completed"

            writer = self._video_writers.pop(camera_id, None)
            if writer is not None:
                writer.release()

            self._start_times.pop(camera_id, None)
            self._frame_counts.pop(camera_id, None)
            self._person_count_peaks.pop(camera_id, None)
            self._segment_counts.pop(camera_id, None)
            self._continuous_save_video.pop(camera_id, None)

            self.storage.update_recording(session)
            self._update_status()
            logger.info(
                "录制完成: id=%s camera=%s frames=%d duration=%.1fs",
                session.id,
                camera_id,
                session.frame_count,
                session.duration_s or 0,
            )
            return session

    def feed_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        detection_result: Any,
        tnlf_data: Dict[str, Any],
    ) -> None:
        """
        从 detector 回调喂入一帧数据。

        注意：此方法在 detector 的后处理线程中调用，不是 async。
        内部使用同步文件 I/O（jsonl append）。
        """
        session = self._sessions.get(camera_id)
        if session is None:
            return

        self._frame_counts[camera_id] = self._frame_counts.get(camera_id, 0) + 1
        frame_idx = self._frame_counts[camera_id]
        timestamp = time.time()

        # 峰值人物数
        person_count = len(detection_result.persons) if hasattr(detection_result, "persons") else 0
        if person_count > self._person_count_peaks.get(camera_id, 0):
            self._person_count_peaks[camera_id] = person_count

        # 写入视频帧
        if session.video_path and frame is not None and frame.size > 0:
            writer = self._video_writers.get(camera_id)
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    session.video_path, fourcc, 15.0, (w, h)
                )
                self._video_writers[camera_id] = writer
            writer.write(frame)

        # 写入关键点（取第一个人，或所有人）
        persons = getattr(detection_result, "persons", [])
        if persons:
            person = persons[0]  # 录制时通常聚焦单个人
            # bbox from YOLO detection (x1, y1, x2, y2)
            bbox = getattr(person, "bbox", None)
            if bbox is not None and isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            keypoints_data = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "track_id": getattr(person, "track_id", ""),
                "keypoints": person.keypoints.tolist()
                if isinstance(getattr(person, "keypoints", None), np.ndarray)
                else getattr(person, "keypoints", []),
                "bbox": bbox,
                "left_hand_landmarks": getattr(person, "left_hand_landmarks", None),
                "right_hand_landmarks": getattr(person, "right_hand_landmarks", None),
                "left_palm_normal": person.left_palm_normal.tolist()
                if isinstance(getattr(person, "left_palm_normal", None), np.ndarray)
                else getattr(person, "left_palm_normal", None),
                "right_palm_normal": person.right_palm_normal.tolist()
                if isinstance(getattr(person, "right_palm_normal", None), np.ndarray)
                else getattr(person, "right_palm_normal", None),
            }
            self.storage.append_keypoints(session.id, keypoints_data)

            # 写入检测输出
            detections_data = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "person_count": len(persons),
                "gesture": getattr(person, "gesture", "none"),
                "gesture_conf": getattr(person, "gesture_conf", 0.0),
                "track_id": getattr(person, "track_id", ""),
            }
            self.storage.append_detections(session.id, detections_data)

        # 写入 TNLF 特征
        if tnlf_data:
            tnlf_record = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "left_wrist_local": _tolist(tnlf_data.get("left_wrist_local")),
                "right_wrist_local": _tolist(tnlf_data.get("right_wrist_local")),
                "left_velocity_mag": tnlf_data.get("left_velocity_mag", 0.0),
                "right_velocity_mag": tnlf_data.get("right_velocity_mag", 0.0),
                "left_theta1": tnlf_data.get("left_theta1", 0.0),
                "left_theta2": tnlf_data.get("left_theta2", 0.0),
                "left_ext_ratio": tnlf_data.get("left_ext_ratio", 0.0),
                "right_theta1": tnlf_data.get("right_theta1", 0.0),
                "right_theta2": tnlf_data.get("right_theta2", 0.0),
                "right_ext_ratio": tnlf_data.get("right_ext_ratio", 0.0),
                "left_tnlf_valid": tnlf_data.get("left_tnlf_valid", False),
                "right_tnlf_valid": tnlf_data.get("right_tnlf_valid", False),
            }
            self.storage.append_tnlf(session.id, tnlf_record)

        # 连续录制自动分段检查
        if session.trigger_mode == "auto_continuous":
            segment_frames = getattr(
                self.config.ai, "datalab_segment_frames", self.SEGMENT_FRAMES_DEFAULT
            )
            if frame_idx >= segment_frames and self._loop is not None:
                logger.info(
                    "连续录制达到分段阈值 %d 帧，自动分段: camera=%s session=%s",
                    segment_frames,
                    camera_id,
                    session.id,
                )
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._auto_split(camera_id), self._loop
                    )
                except Exception as e:
                    logger.warning("提交分段任务失败: %s", e)

    def is_recording(self, camera_id: Optional[str] = None) -> bool:
        """是否正在录制。若指定 camera_id，则检查该摄像头；否则检查是否有任何活跃录制。"""
        if camera_id is not None:
            return camera_id in self._sessions
        return len(self._sessions) > 0

    def get_status(self) -> RecorderStatus:
        """获取当前状态（包含所有活跃会话）。"""
        self._update_status()
        return self._status

    def get_current_session_id(self, camera_id: Optional[str] = None) -> Optional[str]:
        """获取当前录制会话 ID。"""
        if camera_id is not None:
            session = self._sessions.get(camera_id)
            return session.id if session else None
        # 返回第一个活跃会话的 ID
        for session in self._sessions.values():
            return session.id
        return None

    def get_active_camera_ids(self) -> List[str]:
        """获取所有正在录制的摄像头 ID 列表。"""
        return list(self._sessions.keys())

    def stop_by_session_id(self, session_id: str) -> Optional[RecordingSession]:
        """根据会话 ID 停止录制（线程安全版本，供 API 同步调用后转 async）。"""
        for cam_id, session in list(self._sessions.items()):
            if session.id == session_id:
                # 返回会话但不在这里做 async stop，让上层通过 camera_id 调用 async stop
                return session
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _auto_split(self, camera_id: str) -> None:
        """连续录制模式下自动分段：结束当前段并立即开启下一段。"""
        if camera_id not in self._sessions:
            return

        save_video = self._continuous_save_video.get(camera_id, True)
        self._segment_counts[camera_id] = self._segment_counts.get(camera_id, 0) + 1

        # 结束当前段
        await self.stop(camera_id)

        # 无缝开启下一段
        try:
            await self.start(
                camera_id=camera_id,
                trigger_mode="auto_continuous",
                save_video=save_video,
            )
            logger.info(
                "连续录制已自动开启第 %d 段 (camera=%s)",
                self._segment_counts.get(camera_id, 0) + 1,
                camera_id,
            )
        except Exception as e:
            logger.error("自动开启下一段失败 (camera=%s): %s", camera_id, e)

    def _update_status(self) -> None:
        sessions_list: List[Dict[str, Any]] = []
        now = time.time()
        for cam_id, session in self._sessions.items():
            sessions_list.append({
                "session_id": session.id,
                "camera_id": cam_id,
                "trigger_mode": session.trigger_mode,
                "frame_count": self._frame_counts.get(cam_id, 0),
                "duration_s": round(now - self._start_times.get(cam_id, now), 2),
                "person_count_peak": self._person_count_peaks.get(cam_id, 0),
            })

        # 主会话（第一个）用于兼容旧版前端
        primary = sessions_list[0] if sessions_list else {}
        self._status = RecorderStatus(
            is_recording=len(self._sessions) > 0,
            session_id=primary.get("session_id"),
            camera_id=primary.get("camera_id"),
            trigger_mode=primary.get("trigger_mode"),
            frame_count=primary.get("frame_count", 0),
            duration_s=primary.get("duration_s", 0.0),
            person_count_peak=primary.get("person_count_peak", 0),
            segment_count=sum(self._segment_counts.values()),
            sessions=sessions_list,
            total_active=len(self._sessions),
        )


def _tolist(val: Any) -> Any:
    """将 numpy array 转为 list。"""
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
