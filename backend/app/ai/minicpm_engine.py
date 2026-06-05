"""
MiniCPM-V 视频推理引擎

为 Simple+MiniCPM 混合手势识别器提供 GPU 视频推理能力。
模型常驻显存，推理在后台线程异步执行，不阻塞主帧处理流程。
"""

import logging
import os
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MiniCPMEngine:
    """
    MiniCPM-V GPU 推理引擎。

    - 模型加载：transformers AutoModelForImageTextToText + AutoProcessor
    - 常驻显存，支持高并发（通过 ThreadPoolExecutor）
    - 视频输入：将帧序列保存为临时 MP4，通过 messages 传入
    - 结果异步更新，不阻塞调用方
    """

    # 优化 Prompt：聚焦拦车/打招呼意图，放宽姿态限制
    DEFAULT_PROMPT = (
        "These frames are from a roadside surveillance camera mounted on a vehicle. "
        "A person is visible near the road. "
        "Does this person clearly intend to get the vehicle's attention "
        "by raising their arm or waving their hand, "
        "as if hailing a taxi, greeting the driver, or asking the vehicle to stop? "
        "The person may be facing the camera, sideways, or at an angle. "
        "Focus on the intent to interact with the vehicle, not the exact hand orientation. "
        "Do NOT count: scratching head, talking on phone, shading eyes from sun, "
        "stretching, adjusting clothes, or holding objects casually. "
        "Answer only 'yes' or 'no'."
    )

    def __init__(
        self,
        model_path: str,
        prompt: Optional[str] = None,
        max_concurrent: int = 16,
        confidence_threshold: float = 0.6,
        device: str = "cuda",
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> None:
        self.model_path = model_path
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.confidence_threshold = confidence_threshold
        self.device = device

        # 线程池：控制并发推理数
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="minicpm",
        )

        # 结果缓存：track_id -> List[(start_frame_idx, end_frame_idx, gesture, confidence)]
        # 按时间窗口记录，支持 get_result_at(frame_idx) 精确查询
        self._results: Dict[str, List[Tuple[int, int, str, float]]] = {}
        # 提交队列：track_id -> List[(track_id, frames, frame_indices)]
        self._queue: Dict[str, List[Tuple[str, List[np.ndarray], List[int]]]] = {}
        self._pending: set = set()
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

        # 模型 & processor（支持外部传入共享实例，避免多实例并发加载导致模型损坏）
        if model is not None and processor is not None:
            self._model = model
            self._processor = processor
            self._loaded = True
            logger.info("MiniCPMEngine: 使用外部共享模型实例")
        else:
            self._model = None
            self._processor = None
            self._loaded = False
            # 后台常驻：启动时立即加载模型到显存
            self._executor.submit(self._ensure_loaded)

    # ------------------------------------------------------------------
    # 模型加载（线程安全，懒加载）
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._load_lock:
            # 双重检查
            if self._loaded:
                return True
            try:
                from transformers import (
                    AutoModelForImageTextToText,
                    AutoProcessor,
                )

                logger.info("MiniCPM 开始加载模型: %s", self.model_path)
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                # 加载到 cuda:0（YOLO 也在 cuda:0，但 YOLO 只占 ~500MB，24GB 足够两者共用）
                # cuda:1 被宿主机 VLLM 占用 17.9GB，留给 MiniCPM 的空间不够
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                )
                self._model.eval()
                self._loaded = True
                logger.info("MiniCPM 模型加载完成")
                return True
            except Exception as e:
                logger.error("MiniCPM 模型加载失败: %s", e, exc_info=True)
                return False

    # ------------------------------------------------------------------
    # 公共接口：提交推理（不阻塞，使用队列顺序处理）
    # ------------------------------------------------------------------
    def submit(
        self,
        track_id: str,
        frames: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> None:
        """提交视频帧进行异步推理，立即返回。同一 track_id 的多次提交会进入队列顺序执行。

        Args:
            track_id: 跟踪 ID
            frames: 视频帧列表（BGR numpy arrays）
            frame_indices: 各帧对应的全局帧索引，用于结果回填时精确定位
        """
        with self._lock:
            self._queue.setdefault(track_id, []).append(
                (track_id, frames, frame_indices or [])
            )
            if track_id in self._pending:
                return
            self._pending.add(track_id)

        # 在线程池中执行队列处理器
        self._executor.submit(self._process_queue, track_id)

    def get_result(self, track_id: str) -> Tuple[str, float]:
        """获取该 track 的最新推理结果（不阻塞）。向后兼容。"""
        with self._lock:
            windows = self._results.get(track_id)
        if not windows:
            return "none", 0.0
        # 返回最后一个窗口的结果
        _, _, gesture, conf = windows[-1]
        return gesture, conf

    def get_result_at(self, track_id: str, frame_idx: int) -> Tuple[str, float]:
        """获取覆盖指定 frame_idx 的推理窗口结果（不阻塞）。

        从后往前查找，返回最匹配的窗口结果。如果没有窗口覆盖该帧，返回 ("none", 0.0)。
        """
        with self._lock:
            windows = self._results.get(track_id)
        if not windows:
            return "none", 0.0
        for start, end, gesture, conf in reversed(windows):
            if start <= frame_idx <= end:
                return gesture, conf
        return "none", 0.0

    def get_inference_windows(
        self, track_id: str
    ) -> List[Tuple[int, int, str, float]]:
        """获取该 track 的所有推理窗口及其结果。"""
        with self._lock:
            return list(self._results.get(track_id, []))

    def wait_for_result(self, track_id: str, timeout: float = 30.0) -> Tuple[str, float]:
        """阻塞等待该 track 的队列全部处理完成，返回最新结果。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if track_id not in self._pending:
                    windows = self._results.get(track_id)
                    if windows:
                        _, _, gesture, conf = windows[-1]
                        return gesture, conf
                    return "none", 0.0
            time.sleep(0.05)
        logger.warning("MiniCPM wait_for_result[%s] timeout after %.1fs", track_id, timeout)
        with self._lock:
            windows = self._results.get(track_id)
            if windows:
                _, _, gesture, conf = windows[-1]
                return gesture, conf
        return "none", 0.0

    def cleanup_stale(self, active_track_ids: set) -> None:
        """清理不再活跃的 track 的缓存和队列状态。"""
        with self._lock:
            stale = [
                tid for tid in list(self._results.keys())
                if tid not in active_track_ids
            ]
            for tid in stale:
                self._results.pop(tid, None)
                self._queue.pop(tid, None)
                self._pending.discard(tid)

    def reset(self) -> None:
        """重置所有状态（用于测试/调试）。"""
        with self._lock:
            self._results.clear()
            self._queue.clear()
            self._pending.clear()

    # ------------------------------------------------------------------
    # 内部：队列处理器（顺序处理同一 track_id 的所有提交）
    # ------------------------------------------------------------------
    def _process_queue(self, track_id: str) -> None:
        while True:
            with self._lock:
                queue = self._queue.get(track_id, [])
                if not queue:
                    self._pending.discard(track_id)
                    break
                _, frames, frame_indices = queue.pop(0)
            try:
                if not self._ensure_loaded():
                    continue
                gesture, conf = self._sync_infer(track_id, frames)
                # 计算本次推理覆盖的帧索引范围
                if frame_indices:
                    window_start = int(min(frame_indices))
                    window_end = int(max(frame_indices))
                else:
                    window_start = window_end = -1
                with self._lock:
                    self._results.setdefault(track_id, []).append(
                        (window_start, window_end, gesture, conf)
                    )
            except Exception as e:
                logger.error("MiniCPM 推理失败[%s]: %s", track_id, e, exc_info=True)

    def _sync_infer(self, track_id: str, frames: List[np.ndarray]) -> Tuple[str, float]:
        """同步推理：将帧序列直接传入 MiniCPM processor，绕过视频文件 I/O。

        使用 4D numpy array (T, H, W, C) + do_sample_frames=False 避免
        cv2.VideoWriter/VideoCapture 在实验环境下偶发的数十秒阻塞问题。
        """
        if not frames:
            return "none", 0.0

        t0 = time.time()

        # 统一缩放到目标尺寸并堆叠为 4D 数组
        target_h, target_w = 448, 448
        resized_frames = []
        for f in frames:
            if f.shape[:2] != (target_h, target_w):
                f = cv2.resize(f, (target_w, target_h))
            resized_frames.append(f)
        video_array = np.stack(resized_frames)  # (T, H, W, C)
        t1 = time.time()

        try:
            # 1. 构造消息（含视频占位符）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": video_array},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            # 2. apply_chat_template 生成带占位符的文本
            prompt = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

            # 3. processor 处理视频+文本
            inputs = self._processor(
                videos=[video_array],
                text=prompt,
                return_tensors="pt",
                do_sample_frames=False,
            )

            # 4. 移到 GPU 并统一 dtype（模型可能是 BFloat16，processor 输出 Float32）
            device = next(self._model.parameters()).device
            model_dtype = next(self._model.parameters()).dtype
            for k in list(inputs.keys()):
                if hasattr(inputs[k], "to"):
                    if torch.is_floating_point(inputs[k]):
                        inputs[k] = inputs[k].to(device=device, dtype=model_dtype)
                    else:
                        inputs[k] = inputs[k].to(device=device)

            # 5. 生成（限制最大显存占用）
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                )

            # 6. 解码
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            output_text = output_text.strip().lower()
            t2 = time.time()
            logger.info(
                "MiniCPM 推理[%s]: raw='%s' preprocess=%.2fs infer=%.2fs total=%.2fs frames=%d",
                track_id, output_text, t1 - t0, t2 - t1, t2 - t0, len(frames),
            )

            # 解析 yes/no
            is_waving = "yes" in output_text and "no" not in output_text
            conf = 0.85 if is_waving else 0.15
            if is_waving and self.confidence_threshold > 0:
                conf = max(conf, self.confidence_threshold + 0.05)

            return ("waving" if is_waving else "none"), conf
        except Exception as e:
            logger.error("MiniCPM 推理异常[%s]: %s", track_id, e, exc_info=True)
            return "none", 0.0

    @staticmethod
    def _frames_to_temp_video(frames: List[np.ndarray], target_size: int = 448) -> str:
        """将 numpy 帧序列（BGR）保存为临时 MP4 文件，返回路径。

        所有帧统一缩放到 target_size x target_size，避免视频编码后因尺寸不一致
        导致 MiniCPM processor 产生不同的 patch 数，进而触发 tensor size mismatch。
        """
        if not frames:
            raise ValueError("Empty frames")

        h, w = target_size, target_size
        t0 = time.time()
        # 使用 /dev/shm（tmpfs 内存文件系统）避免磁盘 I/O 瓶颈
        fd, path = tempfile.mkstemp(suffix=".mp4", dir="/dev/shm")
        os.close(fd)
        t1 = time.time()

        # 使用 mp4v 编码器（兼容性最好）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 固定 8fps，让模型知道时间节奏
        writer = cv2.VideoWriter(path, fourcc, 8.0, (w, h))
        t2 = time.time()
        if not writer.isOpened():
            writer.release()
            os.remove(path)
            raise RuntimeError("cv2.VideoWriter 打开失败")

        for i, f in enumerate(frames):
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
        t3 = time.time()
        writer.release()
        t4 = time.time()

        # 简单校验：确保写入了预期数量的帧
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            actual_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual_count != len(frames):
                logger.warning(
                    "MiniCPM temp video 帧数不匹配: 期望 %d, 实际 %d", len(frames), actual_count
                )
        t5 = time.time()

        logger.info(
            "_frames_to_temp_video timing: mkstemp=%.3fs writer_open=%.3fs write=%.3fs release=%.3fs verify=%.3fs total=%.3fs frames=%d path=%s",
            t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0, len(frames), path,
        )

        return path
