"""
系统配置模块

使用 dataclass 定义所有配置项，支持通过环境变量覆盖默认值。
所有配置项均可在运行前通过环境变量进行自定义。
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数值，失败时返回默认值。"""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {name} 格式错误，使用默认值 {default}")
        return default


def _env_float(name: str, default: float) -> float:
    """从环境变量读取浮点数值，失败时返回默认值。"""
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {name} 格式错误，使用默认值 {default}")
        return default


def _env_str(name: str, default: str) -> str:
    """从环境变量读取字符串值，不存在时返回默认值。"""
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值，支持 true/1/yes/on；未设置或空串时使用 default。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    val = str(raw).lower().strip()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default


@dataclass
class StreamConfig:
    """视频流相关配置。"""

    # 帧率限制 (FPS)
    fps: int = field(default_factory=lambda: _env_int("STREAM_FPS", 15))
    # 视频宽度 (像素)，高分辨率有利于远处人体再缩放后仍保留细节
    width: int = field(default_factory=lambda: _env_int("STREAM_WIDTH", 1280))
    # 视频高度 (像素)
    height: int = field(default_factory=lambda: _env_int("STREAM_HEIGHT", 720))
    # JPEG压缩质量 (1-100)
    jpeg_quality: int = field(default_factory=lambda: _env_int("JPEG_QUALITY", 85))
    # 帧缓冲队列最大长度
    buffer_size: int = field(default_factory=lambda: _env_int("STREAM_BUFFER_SIZE", 5))
    # 低延迟采集：网络流不节流、使用更小队列以追新帧
    low_latency_capture: bool = field(
        default_factory=lambda: _env_bool("STREAM_LOW_LATENCY", True)
    )
    # WebSocket 推流侧自适应：短边最小/最大（像素）
    adaptive_min_short_side: int = field(
        default_factory=lambda: _env_int("ADAPTIVE_MIN_SHORT_SIDE", 256)
    )
    adaptive_max_short_side: int = field(
        default_factory=lambda: _env_int("ADAPTIVE_MAX_SHORT_SIDE", 960)
    )
    # 自动重连最大重试次数
    reconnect_max_retries: int = field(
        default_factory=lambda: _env_int("STREAM_RECONNECT_RETRIES", 5)
    )
    # 自动重连间隔 (秒)
    reconnect_interval: float = field(
        default_factory=lambda: _env_float("STREAM_RECONNECT_INTERVAL", 3.0)
    )


@dataclass
class AIConfig:
    """AI推理相关配置。"""

    # YOLO11-Pose 模型：x 档为 Ultralytics 最新一代精度最高档
    # 相比 YOLOv8x-Pose，参数量减少约 22%，关键点定位精度提升
    yolo_model: str = field(
        default_factory=lambda: _env_str("YOLO_MODEL", "yolo11x-pose.pt")
    )
    # 模型文件下载保存目录
    model_dir: str = field(
        default_factory=lambda: _env_str("MODEL_DIR", "./models")
    )
    # 推理置信度阈值
    conf_threshold: float = field(
        default_factory=lambda: _env_float("AI_CONF_THRESHOLD", 0.5)
    )
    # 招手动作 - 手腕x坐标移动阈值 (归一化坐标)
    wave_threshold: float = field(
        default_factory=lambda: _env_float("WAVE_THRESHOLD", 0.15)
    )
    # 手势识别历史帧数 (用于检测挥动动作)
    gesture_history_frames: int = field(
        default_factory=lambda: _env_int("GESTURE_HISTORY_FRAMES", 10)
    )
    # 手势识别置信度阈值
    gesture_conf_threshold: float = field(
        default_factory=lambda: _env_float("GESTURE_CONF_THRESHOLD", 0.6)
    )
    # 检测目标最大数量
    max_detections: int = field(
        default_factory=lambda: _env_int("AI_MAX_DETECTIONS", 10)
    )
    # YOLO 推理输入边长（建议 32 倍数；越大远处小人越易检出，越吃显存/算力）
    inference_imgsz: int = field(
        default_factory=lambda: _env_int("AI_INFERENCE_IMGSZ", 640)
    )
    # 是否使用 FP16 半精度推理：true 时推理速度提升约 40~60%，精度损失极小
    inference_half: bool = field(
        default_factory=lambda: _env_bool("AI_INFERENCE_HALF", True)
    )
    # 是否启用MediaPipe手部检测
    enable_hand_detection: bool = field(
        default_factory=lambda: _env_bool("ENABLE_HAND_DETECTION", True)
    )
    # 是否启用 ByteTrack 多目标跟踪（替换轻量 bbox 关联）
    enable_tracking: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TRACKING", True)
    )
    # 手势动作最小持续时间（秒），用于区分真实意图与偶发动作
    gesture_min_duration_s: float = field(
        default_factory=lambda: _env_float("GESTURE_MIN_DURATION_S", 2.5)
    )
    # 时间窗口内手掌朝向画面的最小帧占比
    gesture_palm_facing_ratio: float = field(
        default_factory=lambda: _env_float("GESTURE_PALM_FACING_RATIO", 0.60)
    )
    # 时间窗口内手臂伸直/高举的最小帧占比
    gesture_arm_pose_ratio: float = field(
        default_factory=lambda: _env_float("GESTURE_ARM_POSE_RATIO", 0.50)
    )
    # 运动方向纯度阈值（主方向位移 / 总位移）
    gesture_motion_purity: float = field(
        default_factory=lambda: _env_float("GESTURE_MOTION_PURITY", 0.65)
    )
    # 最小方向反转周期数（2 次反转 = 1 个完整周期）
    gesture_min_cycles: int = field(
        default_factory=lambda: _env_int("GESTURE_MIN_CYCLES", 2)
    )
    # 单个挥动周期的最大允许时长（秒），过滤过慢/非周期动作
    gesture_cycle_max_period_s: float = field(
        default_factory=lambda: _env_float("GESTURE_CYCLE_MAX_PERIOD_S", 1.5)
    )
    # 自然伸直手臂的最小夹角（shoulder-elbow-wrist，单位度）
    gesture_straight_arm_angle: float = field(
        default_factory=lambda: _env_float("GESTURE_STRAIGHT_ARM_ANGLE", 120.0)
    )
    # hailing 判定：手腕最低不得低于肩膀下方 torso_h * ratio
    gesture_hailing_min_height_ratio: float = field(
        default_factory=lambda: _env_float("GESTURE_HAILING_MIN_HEIGHT_RATIO", 0.3)
    )

    # --- 三锁合取机制参数（Torso-Normalized Local Frame）---
    # θ1 角度阈值：手臂抬起最小角度（hip-shoulder-elbow，度）
    gesture_theta1_hailing_min: float = field(
        default_factory=lambda: _env_float("GESTURE_THETA1_HAILING_MIN", 25.0)
    )
    # θ2 角度阈值：手臂伸直最小角度（shoulder-elbow-wrist，度）
    gesture_theta2_straight_min: float = field(
        default_factory=lambda: _env_float("GESTURE_THETA2_STRAIGHT_MIN", 15.0)
    )
    # 手臂伸展比例最小值（|shoulder-wrist| / (|SE|+|EW|)）
    gesture_arm_extension_min: float = field(
        default_factory=lambda: _env_float("GESTURE_ARM_EXTENSION_MIN", 0.10)
    )

    # 姿态锁最小持续帧数
    gesture_pose_min_frames: int = field(
        default_factory=lambda: _env_int("GESTURE_POSE_MIN_FRAMES", 3)
    )
    # 朝向锁最小持续帧数
    gesture_orientation_min_frames: int = field(
        default_factory=lambda: _env_int("GESTURE_ORIENTATION_MIN_FRAMES", 5)
    )
    # 运动锁最小持续帧数
    gesture_motion_min_frames: int = field(
        default_factory=lambda: _env_int("GESTURE_MOTION_MIN_FRAMES", 3)
    )
    # 确认后保持帧数（约 1 秒 @ 15fps）
    gesture_hold_max_frames: int = field(
        default_factory=lambda: _env_int("GESTURE_HOLD_MAX_FRAMES", 15)
    )

    # 朝向锁：法向量与摄像头视线方向夹角阈值（度），放宽以补偿 MediaPipe Z 轴不可靠
    gesture_orientation_lock_angle: float = field(
        default_factory=lambda: _env_float("GESTURE_ORIENTATION_LOCK_ANGLE", 55.0)
    )
    # 朝向锁释放角度（度）
    gesture_orientation_release_angle: float = field(
        default_factory=lambda: _env_float("GESTURE_ORIENTATION_RELEASE_ANGLE", 60.0)
    )

    # 运动锁：FFT 频率范围（Hz），下限 0.35 覆盖年长/缓慢挥手者
    gesture_motion_freq_min: float = field(
        default_factory=lambda: _env_float("GESTURE_MOTION_FREQ_MIN", 0.35)
    )
    gesture_motion_freq_max: float = field(
        default_factory=lambda: _env_float("GESTURE_MOTION_FREQ_MAX", 3.0)
    )
    # 运动锁：最小振幅（torso_units）
    gesture_motion_amp_min: float = field(
        default_factory=lambda: _env_float("GESTURE_MOTION_AMP_MIN", 0.1)
    )
    # 运动锁：速度释放阈值（torso_units/s）
    gesture_motion_speed_min: float = field(
        default_factory=lambda: _env_float("GESTURE_MOTION_SPEED_MIN", 0.05)
    )

    # 面部过滤：硬过滤阈值
    gesture_facing_hard_threshold: float = field(
        default_factory=lambda: _env_float("GESTURE_FACING_HARD_THRESHOLD", 0.25)
    )
    # 面部过滤：软过滤上限
    gesture_facing_soft_threshold: float = field(
        default_factory=lambda: _env_float("GESTURE_FACING_SOFT_THRESHOLD", 0.6)
    )

    # 周期性检测：最小完整周期数
    gesture_period_min_cycles: int = field(
        default_factory=lambda: _env_int("GESTURE_PERIOD_MIN_CYCLES", 2)
    )
    # 周期性检测：最小周期一致性（0-1）
    gesture_period_consistency_min: float = field(
        default_factory=lambda: _env_float("GESTURE_PERIOD_CONSISTENCY_MIN", 0.5)
    )
    # 周期性检测：频率范围（Hz），下限覆盖缓慢挥手
    gesture_period_min_freq: float = field(
        default_factory=lambda: _env_float("GESTURE_PERIOD_MIN_FREQ", 0.35)
    )
    gesture_period_max_freq: float = field(
        default_factory=lambda: _env_float("GESTURE_PERIOD_MAX_FREQ", 3.0)
    )

    # 置信度 EMA 平滑系数（alpha，0-1）
    gesture_ema_alpha: float = field(
        default_factory=lambda: _env_float("GESTURE_EMA_ALPHA", 0.35)
    )
    # 置信度输出阈值
    gesture_confidence_threshold: float = field(
        default_factory=lambda: _env_float("GESTURE_CONFIDENCE_THRESHOLD", 0.55)
    )

    # --- Transformer 手势引擎 ---
    # 手势引擎模式: "triplelock" | "transformer" | "transformer_triplelock" (旧名 "hybrid" 仍兼容)
    gesture_engine: str = field(
        default_factory=lambda: _env_str("GESTURE_ENGINE", "triplelock")
    )

    # --- DataLab 数据实验室 ---
    datalab_recordings_dir: str = field(
        default_factory=lambda: _env_str("DATALAB_RECORDINGS_DIR", "data/datalab/recordings")
    )
    datalab_experiments_dir: str = field(
        default_factory=lambda: _env_str("DATALAB_EXPERIMENTS_DIR", "data/datalab/experiments")
    )
    datalab_auto_gesture_buffer_seconds: float = field(
        default_factory=lambda: _env_float("DATALAB_AUTO_GESTURE_BUFFER_S", 5.0)
    )
    datalab_max_recording_frames: int = field(
        default_factory=lambda: _env_int("DATALAB_MAX_RECORDING_FRAMES", 9000)
    )
    # Transformer 模型路径
    transformer_model_path: str = field(
        default_factory=lambda: _env_str(
            "TRANSFORMER_MODEL_PATH", "/app/models/waving_transformer.pt"
        )
    )
    # Transformer 置信度阈值
    transformer_confidence_threshold: float = field(
        default_factory=lambda: _env_float("TRANSFORMER_CONFIDENCE_THRESHOLD", 0.5)
    )

    # --- MiniCPM 视频手势引擎 ---
    minicpm_model_path: str = field(
        default_factory=lambda: _env_str(
            "MINICPM_MODEL_PATH", "OpenBMB/MiniCPM-V-4.6"
        )
    )
    minicpm_input_size: int = field(
        default_factory=lambda: _env_int("MINICPM_INPUT_SIZE", 448)
    )
    minicpm_max_frames: int = field(
        default_factory=lambda: _env_int("MINICPM_MAX_FRAMES", 16)
    )
    minicpm_buffer_seconds: float = field(
        default_factory=lambda: _env_float("MINICPM_BUFFER_SECONDS", 2.5)
    )
    minicpm_min_frames: int = field(
        default_factory=lambda: _env_int("MINICPM_MIN_FRAMES", 4)
    )
    # 缩短推理间隔，提高视频覆盖度；取消冷却期，持续招手场景不遗漏
    minicpm_infer_interval_s: float = field(
        default_factory=lambda: _env_float("MINICPM_INFER_INTERVAL_S", 0.2)
    )
    minicpm_max_concurrent: int = field(
        default_factory=lambda: _env_int("MINICPM_MAX_CONCURRENT", 16)
    )
    minicpm_confidence_threshold: float = field(
        default_factory=lambda: _env_float("MINICPM_CONFIDENCE_THRESHOLD", 0.6)
    )
    minicpm_cooldown_seconds: float = field(
        default_factory=lambda: _env_float("MINICPM_COOLDOWN_SECONDS", 0.0)
    )
    # MiniCPM 推理 Prompt，可通过环境变量自定义
    minicpm_prompt: str = field(
        default_factory=lambda: _env_str(
            "MINICPM_PROMPT",
            "These frames are from a roadside surveillance camera. "
            "The person in the video is facing the camera. "
            "Is the person raising or waving their hand toward the camera, "
            "as if greeting someone or hailing a vehicle? "
            "Do NOT count scratching head, holding a phone, shielding eyes from sun, "
            "or other non-greeting gestures. "
            "Answer only 'yes' or 'no'.",
        )
    )


@dataclass
class ServerConfig:
    """服务器相关配置。"""

    # 服务监听主机
    host: str = field(default_factory=lambda: _env_str("SERVER_HOST", "0.0.0.0"))
    # 服务监听端口
    port: int = field(default_factory=lambda: _env_int("SERVER_PORT", 8000))
    # 是否启用调试模式
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    # CORS允许的源列表
    cors_origins: list[str] = field(
        default_factory=lambda: _env_str(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")
    )
    # WebSocket推送间隔 (秒)
    ws_push_interval: float = field(
        default_factory=lambda: _env_float("WS_PUSH_INTERVAL", 0.1)
    )
    # 单帧推流预算（毫秒）：超过则降低分辨率/JPEG 质量以换流畅度
    ws_frame_budget_ms: float = field(
        default_factory=lambda: _env_float("WS_FRAME_BUDGET_MS", 800.0)
    )
    # 日志级别
    log_level: str = field(
        default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper()
    )


@dataclass
class CameraConfig:
    """摄像头默认配置。"""

    # 默认4路摄像头配置 (位置 -> 视频源URL映射)
    # 格式: camera_name=rtsp_url|file_path|device_id
    default_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "front": _env_str("CAMERA_FRONT", ""),
            "back": _env_str("CAMERA_BACK", ""),
            "left": _env_str("CAMERA_LEFT", ""),
            "right": _env_str("CAMERA_RIGHT", ""),
        }
    )
    # 最大支持的摄像头数量
    max_cameras: int = field(default_factory=lambda: _env_int("MAX_CAMERAS", 4))


@dataclass
class AppConfig:
    """应用程序总配置，聚合所有子配置。"""

    stream: StreamConfig = field(default_factory=StreamConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)


# 全局配置实例（应用启动时初始化）
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    获取全局配置实例（单例模式）。

    Returns:
        AppConfig: 应用程序配置对象
    """
    global _config
    if _config is None:
        _config = AppConfig()
        logger.info("配置已初始化，当前日志级别: %s", _config.server.log_level)
    return _config


def reload_config() -> AppConfig:
    """
    重新加载配置（用于运行时刷新环境变量）。

    Returns:
        AppConfig: 新的应用程序配置对象
    """
    global _config
    _config = AppConfig()
    logger.info("配置已重新加载")
    return _config
