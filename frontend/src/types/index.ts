// ========== 基础类型 ==========

export type CameraId = string;

export type Protocol = 'rtsp' | 'rtmp' | 'http' | 'webrtc' | 'local' | 'file';

export type Gesture = 'waving' | 'hand_up' | 'checking' | 'none';

// ========== 显示配置 ==========

export interface DisplayConfig {
  order: string[];              // 摄像头显示顺序
  labels: Record<string, string>; // 摄像头显示名称
}

// ========== 关键点类型 ==========

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface Pose {
  keypoints: Keypoint[];
  score: number;
}

// ========== 单人物检测 ==========

export interface PersonDetection {
  bbox: [number, number, number, number];
  confidence: number;
  gesture: Gesture;
  gesture_conf: number;
}

// ========== 检测结果 ==========

export interface DetectionResult {
  camera_id: CameraId;
  person_count: number;
  detections: PersonDetection[];
  best_gesture: Gesture;
  best_gesture_confidence: number;
  inference_ms: number;
  timestamp: number;
}

// ========== 消息类型 ==========

export interface FrameMessage {
  type: 'frame';
  camera_id: CameraId;
  data: string; // base64 encoded image
  timestamp: number;
}

export interface StatusMessage {
  type: 'status';
  camera_id: CameraId;
  online: boolean;
  fps: number;
}

export type WebSocketMessage = FrameMessage | StatusMessage;

// ========== 摄像头配置 ==========

export interface CameraConfig {
  id: CameraId;
  protocol: Protocol;
  source: string;
  enabled: boolean;
  label: string;
}

export type CameraConfigs = Record<string, CameraConfig>;

export type PullMethod = 'webrtc' | 'rtmp';

export interface CameraNameMapping {
  front: string;
  back: string;
  left: string;
  right: string;
}

export interface VehicleCameraProfile {
  id: string;
  name: string;
  vehicleCount: number;
  cameras: CameraNameMapping;
}

export interface CameraProfileBundle {
  id: string;
  name: string;
  defaultPullMethod: PullMethod;
  webrtcPlayerBaseUrl: string;
  webrtcApiBaseUrl: string;
  rtmpBaseUrl: string;
  profiles: VehicleCameraProfile[];
}

export interface CameraProfileDocument {
  version: string;
  bundles: CameraProfileBundle[];
}

// ========== WebSocket Hook 返回类型 ==========

export interface UseWebSocketReturn {
  connected: boolean;
  connecting: boolean;
  lastError: string;
  frames: Record<string, string>;
  detections: Record<string, DetectionResult>;
  fps: number;
  sendMessage: (msg: object) => void;
  reconnect: () => void;
}

// ========== 常量 ==========

export const DEFAULT_CAMERA_LABELS: Record<string, string> = {
  front: '前视摄像头',
  back: '后视摄像头',
  left: '左视摄像头',
  right: '右视摄像头',
};

export const DEFAULT_CAMERA_SHORT_LABELS: Record<string, string> = {
  front: '前视',
  back: '后视',
  left: '左视',
  right: '右视',
};

// 兼容性保留旧常量名
export const CAMERA_LABELS = DEFAULT_CAMERA_LABELS;
export const CAMERA_SHORT_LABELS = DEFAULT_CAMERA_SHORT_LABELS;

export const GESTURE_LABELS: Record<Gesture, string> = {
  waving: '招手',
  hand_up: '举手',
  checking: '检测中',
  none: '无手势',
};

export const PROTOCOL_LABELS: Record<Protocol, string> = {
  rtsp: 'RTSP',
  rtmp: 'RTMP',
  http: 'HTTP',
  webrtc: 'WebRTC',
  local: '本地摄像头',
  file: '视频文件',
};

export const DEFAULT_CAMERA_CONFIGS: CameraConfigs = {
  front: {
    id: 'front',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '前视摄像头',
  },
  back: {
    id: 'back',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '后视摄像头',
  },
  left: {
    id: 'left',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '左视摄像头',
  },
  right: {
    id: 'right',
    protocol: 'rtsp',
    source: '',
    enabled: true,
    label: '右视摄像头',
  },
};

export const DEFAULT_DISPLAY_CONFIG: DisplayConfig = {
  order: ['front', 'left', 'right', 'back'],
  labels: { ...DEFAULT_CAMERA_LABELS },
};

// 默认摄像头配置文档（空模板，请根据实际场景填写）
export const DEFAULT_CAMERA_PROFILE_DOCUMENT: CameraProfileDocument = {
  version: '1.0.0',
  bundles: [],
};

// ========== 骨骼连接定义 ==========

export const SKELETON_CONNECTIONS: [number, number][] = [
  [5, 7],   // left_shoulder -> left_elbow
  [7, 9],   // left_elbow -> left_wrist
  [6, 8],   // right_shoulder -> right_elbow
  [8, 10],  // right_elbow -> right_wrist
  [5, 6],   // left_shoulder -> right_shoulder
  [5, 11],  // left_shoulder -> left_hip
  [6, 12],  // right_shoulder -> right_hip
  [11, 12], // left_hip -> right_hip
];

export const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];
