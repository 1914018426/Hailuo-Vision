# Hailuo Vision — 端侧视觉感知 Agent

面向移动智慧零售的**车载视觉招手交互系统**。从传统"传感器+规则引擎"的被动检测设备，升级为具备**端侧零样本语义理解**能力的视觉 Agent：YOLO11-Pose 关键点检测 + 团队自主设计的 Simple 几何预过滤 + MiniCPM-V 4.6 零样本意图识别，实现"几何规则预过滤 + VLM 意图识别"两级流水线，在端侧 GPU 实时运行。

**核心定位**：端侧视觉感知 Agent，无需云端依赖，离线可运行。已在深圳大运、岗厦北地铁站完成实际场景验证。

[![GitHub](https://img.shields.io/badge/GitHub-1914018426%2FHailuo--Vision-blue)](https://github.com/1914018426/Hailuo-Vision)

---

## 目录

1. [项目概览](#项目概览)
2. [核心指标](#核心指标)
3. [系统架构](#系统架构)
4. [快速开始](#快速开始)
5. [系统特性](#系统特性)
6. [算法详解](#算法详解)
7. [DataLab 实验平台](#datalab-实验平台)
8. [引擎模式对比](#引擎模式对比)
9. [配置说明](#配置说明)
10. [API 接口](#api-接口)
11. [开发模式](#开发模式)
12. [项目结构](#项目结构)
13. [许可证](#许可证)

---

## 项目概览

### 从"被动检测"到"视觉 Agent"的升级

| 维度 | 传统 CV 方案 | 本方案 |
|------|-------------|--------|
| 识别方式 | 几何阈值判定（ wrist_y < elbow_y ） | **零样本语义理解，自然语言描述意图** |
| 相似动作区分 | 无法区分招手 vs 打电话/遮阳 | **MiniCPM-V 4.6 语义级区分** |
| 场景迁移 | 换环境需重新采集、标注、调参 | **自然语言描述即可迁移，开箱即用** |
| 部署方式 | 云端 API，延迟高、成本高 | **端侧离线运行，延迟 <200ms** |

### 技术栈

- **感知层**：YOLO11-Pose（COCO-17 + MediaPipe Hands 21点）+ ByteTrack 多目标跟踪
- **预过滤层**：团队自主设计的 Simple 几何规则引擎（FDV 前臂方向向量 + 腕部高度 + 手臂伸展度）
- **意图识别层**：MiniCPM-V 4.6 零样本视觉理解，滑动窗口投票判定
- **实验平台**：DataLab 内置录制、消融实验、统计分析、可视化报告导出
- **推流层**：WebSocket 实时 MJPEG 推流 + GestureOverlay 手势状态叠加

---

## 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 招手识别准确率 | **88%** | Simple + MiniCPM-V 窗口级投票 |
| 负样本误检率 | **5%** | 相比纯 Simple 引擎降低 72% |
| GPU 资源节省 | **49.3%** | 两级流水线 vs 纯 VLM |
| 端侧响应延迟 | **<200ms** | MiniCPM-V 视频级推理 |
| 消融实验规模 | **4963 帧** | 9 组正样本 + 7 组负样本 |
| 部署验证 | **已验证** | 深圳大运、岗厦北地铁站实际运营 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         视频输入层                                │
│         RTSP / RTMP / HTTP / 本地摄像头 / 本地视频文件               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      YOLO11-Pose 检测                             │
│              COCO-17 关键点 + MediaPipe Hands 21点                │
│                    ByteTrack 多目标跟踪                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Simple 引擎预过滤                             │
│           FDV前臂方向向量 + 腕部高度 + 手臂伸展度                   │
│              快速过滤非招手姿态，降低 VLM 调用频率                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓ 候选帧
┌─────────────────────────────────────────────────────────────────┐
│                   MiniCPM-V 4.6 意图识别                          │
│           候选帧裁剪 448x448 → 零样本语义理解                       │
│           区分招手 vs 打电话 / 遮阳 / 挠头 / 伸展                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    滑动窗口投票 + 结果回填                          │
│         推理冷却 30帧 + 窗口级回填 → 高置信度判定                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                     输出与可视化层                                 │
│    WebSocket 实时推流 + GestureOverlay 手势状态叠加 + DataLab 录制  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU + NVIDIA Container Toolkit（CUDA 12.x 兼容）
- 至少 4GB 可用显存
- 8GB 系统内存（推荐 16GB）

### 1. 克隆项目

```bash
git clone git@github.com:1914018426/Hailuo-Vision.git
cd Hailuo-Vision
```

### 2. 准备模型

首次启动时会自动下载 YOLO 模型。如需离线部署：

```bash
mkdir -p models
wget -O models/yolo11s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

### 3. 配置摄像头

编辑 `docker-compose.yml`：

```yaml
environment:
  - CAMERA_FRONT=rtmp://your-rtmp-server/live/stream
  # 或 RTSP: rtsp://192.168.1.100:554/stream
  # 或本地摄像头: 0
  # 或本地文件: /data/video.mp4
```

### 4. 启动服务

```bash
docker compose up -d
```

首次构建约 3-5 分钟。

### 5. 访问系统

| 入口 | 地址 | 说明 |
|------|------|------|
| Web 界面 | http://localhost:18080 | Nginx 统一代理（推荐） |
| API 文档 | http://localhost:18080/api/docs | Swagger/OpenAPI |
| 后端直连 | http://localhost:8001 | FastAPI 服务 |
| 前端直连 | http://localhost:5173 | React 开发服务器 |

---

## 系统特性

### 视觉感知 Agent

| 特性 | 描述 |
|------|------|
| 零样本意图识别 | MiniCPM-V 4.6 自然语言描述区分招手与打电话、遮阳等相似动作，无需重新训练 |
| 两级流水线 | Simple 几何预过滤 + MiniCPM-V 意图识别，GPU 降低 49.3% |
| 滑动窗口投票 | 连续多帧投票判定，精度从单帧 60% 提升至窗口级 88% |
| 推理冷却机制 | 检测到 waving 后 30 帧内不再重复提交，避免资源浪费 |
| 窗口级回填 | 仅对 MiniCPM 实际推理覆盖的帧窗口回填结果，确保精度指标真实 |
| FDV 前臂方向向量 | 团队自主提出，替代手掌法向量代理，轻量级朝向判定 |

### 视觉识别基础

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | YOLO11s-Pose，640x640 输入，GPU fp16 推理 (~10-15ms/帧) |
| 多目标跟踪 | ByteTrack 跨帧关联，最多 20 人同时检测 |
| 躯干归一化坐标系 | TNLF 消除车辆移动伪运动，同一套参数适用于不同距离目标 |
| 自适应推流 | 根据推理负载动态调节分辨率与 JPEG 质量 |
| 实时日志 | WebSocket 实时日志推送 |

### DataLab 实验平台

| 特性 | 描述 |
|------|------|
| 录制控制 | 手动 / 自动手势触发 / 自动连续三种模式 |
| 引擎对比实验 | 多引擎（Simple / MiniCPM / Hybrid）逐帧对比 |
| 组件消融实验 | 逐一移除组件，评估各模块贡献 |
| 全量实验套件 | 自动运行正样本+负样本完整组合 |
| 统计分析 | Precision / Recall / F1 / 一致率矩阵 / 时序一致性 |
| 图表导出 | SVG 条形图 / 折线图 / 热力图 / 雷达图，支持转 PNG |

---

## 算法详解

### 1. 两级流水线：几何预过滤 + VLM 意图识别

```
Frame Capture
    ↓
YOLO11-Pose 关键点检测 (COCO-17 + MediaPipe Hands 21点)
    ↓
Simple 引擎"姿态门" — FDV前臂方向向量 + 腕部高度 + 手臂伸展度
    ↓
MiniCPM-V 4.6 意图识别 — 候选帧裁剪448x448，零样本语义理解
    ↓
滑动窗口投票判定 + 推理冷却30帧 + 窗口级回填
```

**推理冷却与窗口级回填**：检测到 waving 后 30 帧内不再提交新推理请求；仅对 MiniCPM 实际推理覆盖的帧窗口回填"waving"，窗口外标记为"none"，确保精度指标反映真实推理覆盖。

### 2. 前臂方向向量（FDV）

团队自主提出的轻量级几何特征：

```python
forearm = wrist - elbow                          # 前臂 2D 方向
FDV     = [normalize(forearm).x,
           normalize(forearm).y,
           0.5]                                   # Z 分量为正
```

FDV 描述"手臂朝哪个方向伸出"，而非手掌法向量。在招手场景中，手臂朝前上方伸出时 FDV 的 XY 分量小、Z 主导；走路摆臂时 XY 分量大。这避免了运行耗时的 MediaPipe Hands（每帧两次 CPU 推理，帧率下降 4-8 倍）。

### 3. Torso-Normalized Local Frame（TNLF）

以人体自身为参考系，消除车辆移动干扰：

```
origin      = (left_shoulder + right_shoulder) / 2
e_x         = normalize(right_shoulder - left_shoulder)
e_y         = normalize(mid_hip - origin)
torso_scale = |mid_hip - origin|

wrist_local = (dot(wrist - origin, e_x) / torso_scale,
               dot(wrist - origin, e_y) / torso_scale)
```

所有空间特征使用躯干长度归一化，同一套参数适用于不同距离目标。

### 4. 消融实验与帕累托最优

基于 4963 帧（9 组正样本 + 7 组负样本）的完整消融实验：

| 指标 | Simple | Simple+MiniCPM | 纯 MiniCPM |
|------|--------|----------------|------------|
| 工程综合得分 S_engine | 56.4 | **63.6** | 54.5 |
| 负样本 FPR | 0.18 | **0.05** | — |
| GPU 资源占用 | 极低 | **降低 49.3%** | 高 |
| 端侧响应时间 | — | **<200ms** | — |

**结论**：Simple+MiniCPM 综合最优，实现精度与效率的帕累托最优。

### 5. 自适应推流

根据单帧推理耗时动态调节：

```python
if latency > WS_FRAME_BUDGET_MS:           # 超时（默认 60ms）
    short_side -= 12px                      # 降低分辨率
    jpeg_quality -= 2                       # 降低质量
elif latency < WS_FRAME_BUDGET_MS * 0.38:   # 余量充足
    short_side += 8px
    jpeg_quality += 1
```

分辨率范围：320~480px 短边，JPEG 质量 65，单帧约 80KB。

---

## DataLab 实验平台

### 核心 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/datalab/recordings/start` | 开始录制 |
| POST | `/api/datalab/recordings/{id}/stop` | 停止录制 |
| POST | `/api/datalab/experiments/start` | 启动消融实验 |
| GET | `/api/datalab/experiments/{id}/progress` | 实验进度 |
| GET | `/api/datalab/experiments/{id}/report` | 分析报告 |
| GET | `/api/datalab/charts/{chart_type}` | 获取图表 |
| POST | `/api/datalab/import-video` | 导入视频素材 |

### 前端界面

- **DataLabPage**: 录制控制与实验列表
- **AnalysisDashboard**: 统计卡片与可视化图表
- **AblationPanel**: 消融实验配置与结果展示

---

## 引擎模式对比

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `simple-minicpm`（推荐） | Simple 预过滤 + MiniCPM-V 意图识别 | 平衡精度与效率，端侧部署 |
| `simple` | 纯 Simple 几何规则引擎 | 极低 GPU 占用，快速原型 |
| `minicpm` | 纯 MiniCPM-V 零样本识别 | 最高精度，GPU 占用高 |
| `simple-transformer` | Transformer 时序 + Simple 后滤 | 高精度时序识别 |

---

## 配置说明

所有参数通过 `docker-compose.yml` 环境变量配置。

### Agent 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ENABLE_MINICPM_V` | `true` | MiniCPM-V 4.6 意图识别开关 |
| `GESTURE_ENGINE` | `simple-minicpm` | 手势引擎模式 |

### AI 推理

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `YOLO_MODEL` | `yolo11s-pose.pt` | 姿态检测模型 |
| `AI_INFERENCE_IMGSZ` | `640` | YOLO 输入分辨率 |
| `AI_CONF_THRESHOLD` | `0.35` | 人体检测置信度阈值 |
| `ENABLE_TRACKING` | `true` | ByteTrack 跟踪开关 |

### 视频流

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STREAM_FPS` | `15` | 采集/推流目标帧率 |
| `CAMERA_FRONT` | *(必填)* | 视频源地址 |
| `CAMERA_BACK` | — | 后方摄像头 |
| `CAMERA_LEFT` | — | 左侧摄像头 |
| `CAMERA_RIGHT` | — | 右侧摄像头 |

### 推流自适应

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WS_FRAME_BUDGET_MS` | `60` | 单帧推理+编码预算（ms） |
| `JPEG_QUALITY` | `65` | MJPEG 编码质量 |
| `ADAPTIVE_MIN_SHORT_SIDE` | `320` | 自适应最小短边 |
| `ADAPTIVE_MAX_SHORT_SIDE` | `480` | 自适应最大短边 |

---

## API 接口

### REST API

完整 API 文档通过 Swagger UI 提供：http://localhost:18080/api/docs

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cameras` | 列出所有摄像头 |
| POST | `/api/cameras` | 添加摄像头 |
| POST | `/api/cameras/{id}/start` | 启动视频流 |
| POST | `/api/cameras/{id}/stop` | 停止视频流 |
| GET | `/api/health` | 健康检查 |

### WebSocket 接口

| 路径 | 说明 | 消息格式 |
|------|------|---------|
| `/ws/video` | 视频流 + 检测结果 | binary MJPEG + JSON |
| `/ws/logs` | 实时日志推送 | JSON {level, message, timestamp} |

**WebSocket 视频流协议**：

```
[2 bytes: JPEG length (big-endian)] [JPEG data] [JSON metadata]
```

---

## 开发模式

### 前端

```bash
cd frontend
npm install
npm run dev
```

开发服务器运行在 http://localhost:5173。

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 项目结构

```
.
├── docker-compose.yml          # Docker Compose 编排
├── nginx.conf                  # Nginx 反向代理
├── README.md
├── backend/                    # FastAPI 后端
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── entrypoint.sh
│   └── app/
│       ├── main.py               # FastAPI 入口
│       ├── config.py             # 配置中心
│       ├── ai/
│       │   ├── detector.py       # YOLO + ByteTrack + 绘制
│       │   ├── gesture.py        # 手势引擎（Simple / MiniCPM / Hybrid）
│       │   ├── minicpm_engine.py # MiniCPM-V 端侧推理引擎
│       │   ├── local_frame.py    # TNLF 局部参考系
│       │   ├── facing.py         # 面部过滤
│       │   ├── slerp.py          # 法向量平滑
│       │   ├── iri.py            # IRI 意图刚性指数
│       │   ├── bytetrack.yaml    # ByteTrack 配置
│       │   └── transformer/      # Transformer 模型定义与训练
│       ├── api/
│       │   ├── routes.py         # REST API
│       │   ├── ws.py             # WebSocket 视频推流
│       │   └── logs.py           # WebSocket 日志广播
│       ├── stream/
│       │   ├── handler.py        # 视频流解码
│       │   └── manager.py        # 多路流管理
│       └── datalab/
│           ├── api.py            # DataLab REST API
│           ├── models.py         # Pydantic 数据模型
│           ├── recorder.py       # 录制管理器
│           ├── ablation.py       # 消融实验运行器
│           ├── analyzer.py       # 统计分析器
│           ├── charts.py         # SVG/PNG 图表生成
│           └── video_importer.py # 视频导入器
├── frontend/                   # React + Vite + Tailwind
│   ├── Dockerfile
│   ├── nginx-default.conf
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── VideoPanel.tsx
│       │   ├── VideoGrid.tsx
│       │   ├── GestureOverlay.tsx
│       │   ├── StatusBar.tsx
│       │   ├── LogPanel.tsx
│       │   └── CameraConfig.tsx
│       ├── hooks/
│       │   ├── useWebSocket.ts
│       │   ├── useLogWebSocket.ts
│       │   └── useCameraConfig.ts
│       └── datalab/
│           ├── DataLabPage.tsx
│           ├── AnalysisDashboard.tsx
│           ├── AblationPanel.tsx
│           └── HighlightCard.tsx
├── models/                     # 模型持久化目录
│   ├── yolo11s-pose.pt
│   └── transformer/
│       └── waving_transformer_real.pt
└── data/                       # 数据目录
    ├── processed/
    │   └── real_data_seq45.npz
    └── datalab/                # DataLab 录制与实验结果
```

---

## 许可证

MIT License
