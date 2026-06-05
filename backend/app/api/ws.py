"""
WebSocket通信模块

提供实时视频流推送和检测结果传输功能。
每路摄像头独立asyncio任务推送MJPEG视频帧，
同时发送AI检测方向决策结果。

消息格式:
    {
        "camera_id": "front",
        "frame": "<base64编码的JPEG图像>",
        "direction": "front",
        "confidence": 0.85,
        "detections": [...],
        "timestamp": 1703000000.000
    }
"""

import asyncio
import logging
import base64
import json
import struct
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_config
from app.stream.manager import StreamManager
from app.ai.detector import PoseDetector, DetectionResult

logger = logging.getLogger(__name__)

# GPU JPEG 编码（nvjpeg via torchvision），实测 720p 单帧 0.17ms，比 cv2.imencode 快 ~30 倍
_GPU_JPEG_AVAILABLE = False
try:
    from torchvision.io import encode_jpeg as _tv_encode_jpeg
    if torch.cuda.is_available():
        _probe = torch.zeros((3, 16, 16), dtype=torch.uint8, device="cuda")
        _ = _tv_encode_jpeg(_probe, quality=75)
        _GPU_JPEG_AVAILABLE = True
        logger.info("GPU JPEG 编码已启用 (nvjpeg via torchvision)")
except Exception as _e:
    logger.warning("GPU JPEG 不可用，回退 CPU cv2.imencode: %s", _e)

# YOLO 推理在 _DETECTOR_INFER_LOCK 内串行执行（避免并发 forward 抖动）。
# CPU 后处理（绘制/手势/编码）放到多 worker 池并发，让 4 路摄像头真正并行。
_CPU_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu_pipeline")

# 创建WebSocket路由器
router = APIRouter()

# 全局引用（在main.py中初始化时注入）
_stream_manager: Optional[StreamManager] = None
_detector: Optional[PoseDetector] = None


def set_stream_manager(manager: StreamManager) -> None:
    """设置全局流管理器实例。"""
    global _stream_manager
    _stream_manager = manager


def set_detector(detector: PoseDetector) -> None:
    """设置全局检测器实例。"""
    global _detector
    _detector = detector


class ConnectionManager:
    """
    WebSocket连接管理器

    管理所有WebSocket客户端连接，支持多客户端同时接收视频流。
    每个摄像头对应一个独立的广播通道。
    """

    def __init__(self) -> None:
        """初始化连接管理器。"""
        # 所有活跃连接: {websocket: set(camera_ids)}
        self._connections: Dict[WebSocket, Set[str]] = {}
        # 活跃推流任务: {camera_id: asyncio.Task}
        self._tasks: Dict[str, asyncio.Task] = {}
        # 当前连接订阅的摄像头
        self._subscriptions: Dict[WebSocket, Set[str]] = {}
        # Per-ws latest-only 发送槽位与唤醒 event，根治 4 路并发 send_bytes
        # 在同一 TCP 流上 send buffer 累积 ~3s 的端到端延迟
        self._latest_payloads: Dict[WebSocket, Dict[str, bytes]] = {}
        self._payload_events: Dict[WebSocket, asyncio.Event] = {}
        self._sender_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(
        self, websocket: WebSocket, camera_ids: Optional[List[str]] = None
    ) -> None:
        """
        接受新的WebSocket连接。

        Args:
            websocket: WebSocket连接对象
            camera_ids: 客户端请求的摄像头ID列表，None表示全部
        """
        await websocket.accept()
        subscribed = set(camera_ids) if camera_ids else set()
        self._connections[websocket] = subscribed
        self._subscriptions[websocket] = subscribed
        self._latest_payloads[websocket] = {}
        self._payload_events[websocket] = asyncio.Event()
        self._sender_tasks[websocket] = asyncio.create_task(
            self._sender_loop(websocket),
            name=f"ws_sender_{id(websocket)}",
        )

        logger.info(
            "WebSocket客户端已连接: %s, 订阅摄像头: %s",
            websocket.client.host if websocket.client else "unknown",
            subscribed if subscribed else "全部",
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """
        断开WebSocket连接。

        Args:
            websocket: WebSocket连接对象
        """
        self._connections.pop(websocket, None)
        self._subscriptions.pop(websocket, None)
        self._latest_payloads.pop(websocket, None)
        self._payload_events.pop(websocket, None)
        sender = self._sender_tasks.pop(websocket, None)
        # 不能取消 current_task：sender 中 send_bytes 失败时会自调 disconnect
        if sender is not None and not sender.done() and sender is not asyncio.current_task():
            sender.cancel()
        logger.info("WebSocket客户端已断开")

        # 检查是否还有客户端订阅某个摄像头，没有则停止推流任务
        self._cleanup_tasks()

    def _cleanup_tasks(self) -> None:
        """清理没有客户端订阅的推流任务。"""
        # 空订阅集合表示“订阅全部摄像头”，此时不应清理任务
        if any(len(cameras) == 0 for cameras in self._subscriptions.values()):
            return

        # 收集所有仍被订阅的摄像头
        all_subscribed: Set[str] = set()
        for cameras in self._subscriptions.values():
            all_subscribed.update(cameras)

        # 取消没有订阅的任务
        for camera_id in list(self._tasks.keys()):
            if camera_id not in all_subscribed and self._tasks[camera_id]:
                self._tasks[camera_id].cancel()
                self._tasks.pop(camera_id, None)
                logger.info("摄像头 %s 推流任务已取消（无客户端订阅）", camera_id)

    async def send_frame(
        self, websocket: WebSocket, data: Dict
    ) -> None:
        """
        向单个客户端发送帧数据。

        Args:
            websocket: WebSocket连接对象
            data: 帧数据字典
        """
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug("向客户端发送数据失败: %s", str(e))
            raise

    async def broadcast_frame(self, camera_id: str, data: Dict) -> None:
        """
        向所有订阅了指定摄像头的客户端广播帧数据。

        Args:
            camera_id: 摄像头标识
            data: 帧数据字典
        """
        targets: List[WebSocket] = []
        for ws, cameras in list(self._connections.items()):
            if cameras and camera_id not in cameras:
                continue
            targets.append(ws)

        if not targets:
            return

        async def _send(ws: WebSocket) -> Tuple[WebSocket, bool]:
            try:
                await ws.send_json(data)
                return (ws, True)
            except Exception:
                return (ws, False)

        results = await asyncio.gather(*(_send(ws) for ws in targets))
        for ws, ok in results:
            if not ok:
                self.disconnect(ws)

    async def broadcast_binary_frame(
        self, camera_id: str, header: Dict, jpeg_bytes: bytes
    ) -> None:
        """
        向订阅客户端广播二进制帧（消除前端 base64 + JSON.parse 大字符串瓶颈）。

        载荷格式: [4B BE uint32: header_len][header JSON UTF-8][JPEG bytes]

        非阻塞写入：每个 ws 的 per-camera 槽位被新载荷直接覆盖，
        实际 send_bytes 由 _sender_loop 串行消费。这避免 4 路并发 send_bytes
        在内核 send buffer 累积导致 ~3s 端到端排队延迟。
        """
        targets: List[WebSocket] = []
        for ws, cameras in list(self._connections.items()):
            if cameras and camera_id not in cameras:
                continue
            targets.append(ws)

        if not targets:
            return

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = struct.pack(">I", len(header_bytes)) + header_bytes + jpeg_bytes

        for ws in targets:
            slots = self._latest_payloads.get(ws)
            if slots is None:
                continue
            # 旧帧直接被覆盖 — 客户端永远拿到最新帧，落后部分自动丢弃
            slots[camera_id] = payload
            event = self._payload_events.get(ws)
            if event is not None:
                event.set()

    async def _sender_loop(self, websocket: WebSocket) -> None:
        """单 WebSocket 串行 sender。

        每次唤醒：从该 ws 的所有 camera 槽位中取一帧最新载荷发出，
        发送过程中新到的帧只更新槽位，不排队。`send_bytes` 完成后回到 await
        让出控制权，新帧的 `event.set()` 会在下次循环触发再次发送。
        """
        try:
            while True:
                event = self._payload_events.get(websocket)
                if event is None:
                    return
                await event.wait()
                event.clear()
                slots = self._latest_payloads.get(websocket)
                if not slots:
                    continue
                # 取快照后立刻清空槽位 — 还没发完时新帧覆盖旧帧也不影响下一轮
                pending = list(slots.items())
                slots.clear()
                for _camera_id, payload in pending:
                    try:
                        await websocket.send_bytes(payload)
                    except Exception as e:
                        logger.debug("ws sender send_bytes 失败: %s", e)
                        self.disconnect(websocket)
                        return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("ws sender 异常退出: %s", e)
            self.disconnect(websocket)

    @property
    def connection_count(self) -> int:
        """当前连接数量。"""
        return len(self._connections)

    def get_subscribed_cameras(self) -> Set[str]:
        """获取所有客户端订阅的摄像头集合。"""
        all_cameras: Set[str] = set()
        for cameras in self._subscriptions.values():
            all_cameras.update(cameras)
        return all_cameras


# 全局连接管理器实例
connection_manager = ConnectionManager()

# 每路摄像头推流自适应状态：短边像素、JPEG 质量
_push_adapt_state: Dict[str, Dict[str, int]] = {}

# 多路推流共享同一 PoseDetector：串行化 GPU 推理，避免并发 forward 抖动/报错
_DETECTOR_INFER_LOCK = threading.Lock()


def encode_frame_to_base64(
    frame: np.ndarray, quality: int = 85
) -> str:
    """
    将OpenCV图像编码为Base64字符串。

    优先使用 GPU (nvjpeg) 编码，失败时回退 cv2.imencode。

    Args:
        frame: OpenCV图像 (BGR格式)
        quality: JPEG编码质量 (1-100)

    Returns:
        Base64编码的JPEG图像字符串
    """
    raw = encode_frame_to_jpeg_bytes(frame, quality=quality)
    return base64.b64encode(raw).decode("utf-8") if raw else ""


def encode_frame_to_jpeg_bytes(
    frame: np.ndarray, quality: int = 85
) -> bytes:
    """
    将OpenCV图像编码为原始JPEG字节（不做 base64）。

    供二进制 WebSocket 推流使用 — 省去 base64 编/解码与 JSON.parse 大字符串的主线程开销。
    """
    if _GPU_JPEG_AVAILABLE:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(
                "cuda", non_blocking=True
            )
            encoded = _tv_encode_jpeg(tensor, quality=int(quality))
            return encoded.cpu().numpy().tobytes()
        except Exception as e:
            logger.debug("GPU JPEG 编码失败回退 CPU: %s", e)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        return b""
    return encoded.tobytes()


def _resize_to_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
    """将图像缩放到短边不超过 short_side（不放大）。"""
    h, w = frame.shape[:2]
    m = min(h, w)
    if m <= 0 or short_side <= 0:
        return frame
    if m <= short_side:
        return frame
    scale = short_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _get_push_adapt(camera_id: str) -> Dict[str, int]:
    """懒初始化每路摄像头的自适应分辨率与 JPEG 质量。"""
    if camera_id not in _push_adapt_state:
        cfg = get_config()
        lo = max(32, int(cfg.stream.adaptive_min_short_side))
        hi = max(lo, int(cfg.stream.adaptive_max_short_side))
        init_short = min(cfg.stream.width, cfg.stream.height, hi)
        init_short = max(lo, init_short)
        _push_adapt_state[camera_id] = {
            "short": init_short,
            "quality": int(cfg.stream.jpeg_quality),
        }
    return _push_adapt_state[camera_id]


def _process_and_encode_frame(
    detector: PoseDetector,
    work_frame: np.ndarray,
    camera_id: str,
    quality: int,
) -> Tuple[bytes, DetectionResult]:
    """在后台线程中完成检测 + JPEG。

    GPU 推理（YOLO）全局串行化，CPU 后处理（MediaPipe/gesture/绘制）在锁外并行。
    JPEG 编码为纯 CPU，也在锁外执行。返回 raw JPEG bytes 直接给二进制 WebSocket。
    """
    # Phase 1: GPU 推理由全局锁保护（仅 YOLO + ByteTrack ~22ms）
    with _DETECTOR_INFER_LOCK:
        persons = detector.detect_persons(work_frame, camera_id=camera_id)
    # Phase 2: CPU 后处理在锁外执行，多路摄像头可并行
    annotated_frame, detection_result = detector.process_persons(
        work_frame, persons, camera_id=camera_id
    )
    # JPEG 编码为纯 CPU 操作，不占用 GPU
    jpeg_bytes = encode_frame_to_jpeg_bytes(annotated_frame, quality=quality)
    return jpeg_bytes, detection_result


async def camera_push_task(camera_id: str) -> None:
    """
    单个摄像头的推流任务。

    持续获取帧、执行AI检测、编码并广播给所有订阅的客户端。
    每个摄像头运行在独立的asyncio任务中。

    Args:
        camera_id: 摄像头唯一标识
    """
    config = get_config()
    manager = _stream_manager

    if manager is None:
        logger.error("推流任务缺少流管理器实例")
        return

    push_interval = config.server.ws_push_interval
    budget_ms = float(config.server.ws_frame_budget_ms)
    base_quality = int(config.stream.jpeg_quality)
    min_short = max(32, int(config.stream.adaptive_min_short_side))
    max_short = max(min_short, int(config.stream.adaptive_max_short_side))

    logger.info("摄像头 %s 推流任务已启动", camera_id)

    try:
        while True:
            loop_start = time.perf_counter()
            # 只要还有连接就保持推流；具体摄像头过滤由 broadcast_frame 处理
            if connection_manager.connection_count == 0:
                logger.info("摄像头 %s 无客户端订阅，推流任务退出", camera_id)
                break

            detector = _detector
            if detector is None:
                await asyncio.sleep(0.25)
                continue

            # 排空队列，只保留最新一帧，降低端到端延迟
            frame: Optional[np.ndarray] = None
            while True:
                nxt = manager.get_frame_nowait(camera_id)
                if nxt is None:
                    break
                frame = nxt
            if frame is None:
                await asyncio.sleep(push_interval)
                continue

            state = _get_push_adapt(camera_id)
            work_frame = _resize_to_short_side(frame, int(state["short"]))

            # 检测与编码放到单线程执行器串行执行，避免 GIL 争抢。
            # asyncio.to_thread 默认线程池的 4 并发实测比串行慢 4-8 倍。
            try:
                loop = asyncio.get_running_loop()
                jpeg_bytes, detection_result = await loop.run_in_executor(
                    _CPU_EXECUTOR,
                    _process_and_encode_frame,
                    detector,
                    work_frame,
                    camera_id,
                    int(state["quality"]),
                )
            except Exception as e:
                logger.error("AI检测或编码失败 [%s]: %s", camera_id, str(e))
                await asyncio.sleep(push_interval)
                continue

            if not jpeg_bytes:
                await asyncio.sleep(push_interval)
                continue

            # 仅按「推理+编码」耗时调节分辨率，避免把网络发送算进卡顿
            compute_ms = (time.perf_counter() - loop_start) * 1000
            if compute_ms > budget_ms:
                state["short"] = max(min_short, int(state["short"]) - 12)
                state["quality"] = max(50, int(state["quality"]) - 2)
            elif compute_ms < budget_ms * 0.38:
                state["short"] = min(max_short, int(state["short"]) + 8)
                state["quality"] = min(base_quality, int(state["quality"]) + 1)

            # 元数据 header — 不含 frame，JPEG 走二进制载荷
            detections_data = [
                {
                    "bbox": person.bbox,
                    "confidence": round(person.confidence, 4),
                    "gesture": person.gesture,
                    "gesture_conf": round(person.gesture_conf, 4),
                }
                for person in detection_result.persons
            ]

            # DEBUG: 打印发送的 gesture 值
            for d in detections_data:
                logger.info("WS send gesture=%s conf=%.2f", d["gesture"], d["gesture_conf"])

            header = {
                "type": "frame",
                "camera_id": camera_id,
                "detections": detections_data,
                "person_count": len(detection_result.persons),
                "inference_ms": round(detection_result.inference_time_ms, 2),
                "timestamp": time.time(),
            }

            # 二进制广播：[4B header_len][header JSON][JPEG]
            await connection_manager.broadcast_binary_frame(
                camera_id, header, jpeg_bytes
            )

            # 控制推送频率：扣除已耗时，避免固定 sleep 叠加延迟
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, push_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("摄像头 %s 推流任务已取消", camera_id)
        raise  # 重新抛出，让 finally 执行
    except Exception as e:
        logger.error("摄像头 %s 推流任务异常: %s", camera_id, str(e))
    finally:
        # 关键修复：任务退出时必须清理 _tasks 中的自身引用，
        # 否则新客户端连接时不会创建新的推送任务，导致画面永久卡住
        task = connection_manager._tasks.get(camera_id)
        if task is not None and task is asyncio.current_task():
            connection_manager._tasks.pop(camera_id, None)
            logger.info("摄像头 %s 推流任务引用已清理", camera_id)


@router.websocket("/ws/video")
async def video_websocket(websocket: WebSocket) -> None:
    """
    WebSocket视频流端点。

    客户端连接后可指定订阅的摄像头ID列表。
    服务端为每个摄像头启动独立的推流任务，
    实时推送JPEG编码的图像帧和AI检测结果。

    连接方式:
        ws://host:port/ws/video?cameras=front,back,left,right

    消息格式 (服务端发送):
        {
            "camera_id": "front",
            "frame": "<base64_jpeg>",
            "direction": "front",
            "confidence": 0.85,
            "detections": [...],
            "timestamp": 1703000000.000
        }
    """
    # 从查询参数获取订阅的摄像头列表
    query_params = dict(websocket.query_params)
    camera_param = query_params.get("cameras", "")
    requested_cameras: Optional[List[str]] = None
    if camera_param:
        requested_cameras = [c.strip() for c in camera_param.split(",") if c.strip()]

    # 接受连接
    await connection_manager.connect(websocket, requested_cameras)

    try:
        # 获取需要订阅的摄像头
        manager = _stream_manager
        if manager is None:
            await websocket.close(code=1011, reason="流管理器未初始化")
            return

        # 确定实际要推送的摄像头列表
        if requested_cameras:
            target_cameras = [
                cid for cid in requested_cameras if manager.has_camera(cid)
            ]
        else:
            target_cameras = manager.camera_ids

        # 为每个摄像头启动推流任务
        for camera_id in target_cameras:
            if camera_id not in connection_manager._tasks:
                task = asyncio.create_task(
                    camera_push_task(camera_id),
                    name=f"push_{camera_id}",
                )
                connection_manager._tasks[camera_id] = task
                logger.info("摄像头 %s 推流任务已创建", camera_id)

        # 保持连接活跃，处理客户端消息
        while True:
            try:
                # 接收客户端消息（可支持控制命令）
                # 关键修复：添加 30s 超时，避免静默断开的连接永远挂起
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                message = json.loads(data)
                action = message.get("action", "")

                if action == "ping":
                    # NTP 风格回包：用客户端 send_t + 服务端 now 让前端计算时钟偏差
                    await websocket.send_json({
                        "type": "pong",
                        "client_time": message.get("client_time"),
                        "server_time": time.time(),
                    })
                elif action == "subscribe":
                    # 动态订阅摄像头
                    new_cameras = message.get("cameras", [])
                    if isinstance(new_cameras, str):
                        new_cameras = [new_cameras]

                    for cid in new_cameras:
                        if manager.has_camera(cid):
                            connection_manager._subscriptions[websocket].add(cid)
                            if cid not in connection_manager._tasks:
                                task = asyncio.create_task(
                                    camera_push_task(cid),
                                    name=f"push_{cid}",
                                )
                                connection_manager._tasks[cid] = task
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "cameras": list(
                                connection_manager._subscriptions[websocket]
                            ),
                        }
                    )
                elif action == "unsubscribe":
                    # 取消订阅
                    remove_cameras = message.get("cameras", [])
                    if isinstance(remove_cameras, str):
                        remove_cameras = [remove_cameras]
                    for cid in remove_cameras:
                        connection_manager._subscriptions[websocket].discard(cid)
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "cameras": list(
                                connection_manager._subscriptions[websocket]
                            ),
                        }
                    )

            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                logger.warning("收到无效的JSON消息")

    except WebSocketDisconnect:
        logger.info("WebSocket客户端断开连接")
    except Exception as e:
        logger.error("WebSocket连接异常: %s", str(e))
    finally:
        connection_manager.disconnect(websocket)
