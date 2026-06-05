import { useState, useRef, useCallback, useEffect } from 'react';
import type {
  DetectionResult,
  FrameMessage,
  StatusMessage,
  WebSocketMessage,
  UseWebSocketReturn,
} from '@/types';

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_INTERVAL = 3000;
const HEARTBEAT_INTERVAL = 15000;
const FRAME_TIMEOUT_MS = 20000;

export function useWebSocket(): UseWebSocketReturn {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [lastError, setLastError] = useState('');
  const [frames, setFrames] = useState<Record<string, string>>({});
  const [detections, setDetections] = useState<Record<string, DetectionResult>>({});
  const [fps, setFps] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 每路摄像头独立 bucket，避免多路求和变成各路 FPS 之和
  const frameTimestampsRef = useRef<Record<string, number[]>>({});
  const connectingRef = useRef(false);
  const lastFrameTimeRef = useRef(0);

  // rAF 合并：高频 ws.onmessage 写入 ref，rAF 周期内只 setState 一次。
  // 没有这层，4 路 × ~20 fps = 160 次/秒 setState，React reconciler 会卡死，
  // 表现为播放几帧后画面冻结
  const pendingFramesRef = useRef<Record<string, string>>({});
  const pendingDetectionsRef = useRef<Record<string, DetectionResult>>({});
  const dirtyRef = useRef(false);
  const rafIdRef = useRef<number | null>(null);
  // 已注入到 React state 的 blob URL — flush 后立刻 revoke 旧 URL，避免内存堆积
  const liveBlobUrlsRef = useRef<Record<string, string>>({});
  // 端到端延迟探针：header.timestamp（服务端打戳，秒）与浏览器收到时刻的差，1s 一次 p50/p95
  const lagSamplesRef = useRef<number[]>([]);
  // NTP 风格 ping/pong 算出的 server-vs-browser 时钟偏差（ms）。
  // 减去它，[ws-lag] 才是真实管道延迟而非时钟偏差幻影。
  const serverOffsetMsRef = useRef<number>(0);

  const flushPending = useCallback(() => {
    rafIdRef.current = null;
    if (!dirtyRef.current) return;
    dirtyRef.current = false;
    const incomingFrames = pendingFramesRef.current;
    pendingFramesRef.current = {};
    const incomingDetections = pendingDetectionsRef.current;
    pendingDetectionsRef.current = {};
    // 释放被替换掉的旧 blob URL
    for (const cid of Object.keys(incomingFrames)) {
      const oldUrl = liveBlobUrlsRef.current[cid];
      if (oldUrl && oldUrl.startsWith('blob:')) {
        try { URL.revokeObjectURL(oldUrl); } catch { /* noop */ }
      }
      liveBlobUrlsRef.current[cid] = incomingFrames[cid];
    }
    setFrames((prev) => ({ ...prev, ...incomingFrames }));
    setDetections((prev) => ({ ...prev, ...incomingDetections }));
  }, []);

  const scheduleFlush = useCallback(() => {
    dirtyRef.current = true;
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(flushPending);
    }
  }, [flushPending]);

  const updateFps = useCallback(() => {
    const now = Date.now();
    const buckets = frameTimestampsRef.current;
    const active: number[] = [];
    for (const cid of Object.keys(buckets)) {
      buckets[cid] = buckets[cid].filter((t) => now - t < 1000);
      if (buckets[cid].length > 0) {
        active.push(buckets[cid].length);
      }
    }
    if (active.length === 0) {
      setFps(0);
    } else {
      // 取各路流的平均值，而非总和
      const avg = active.reduce((a, b) => a + b, 0) / active.length;
      setFps(Math.round(avg));
    }
  }, []);

  const getWebSocketUrl = useCallback((): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/video`;
  }, []);

  const connect = useCallback(() => {
    if (connectingRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    connectingRef.current = true;
    setConnecting(true);

    try {
      const url = getWebSocketUrl();
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setConnecting(false);
        setLastError('');
        reconnectAttemptsRef.current = 0;
        connectingRef.current = false;
        lastFrameTimeRef.current = Date.now();

        // 立即发一次 ping 触发时钟偏差测量，无需等 15s 心跳
        ws.send(JSON.stringify({ action: 'ping', client_time: Date.now() / 1000 }));

        heartbeatTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'ping', client_time: Date.now() / 1000 }));
          }
        }, HEARTBEAT_INTERVAL);
      };

      // 把 header 中的检测元数据归一化成 DetectionResult；与历史路径共享。
      const ingestHeader = (raw: {
        camera_id?: string;
        person_count?: number;
        inference_ms?: number;
        detections?: Array<{
          bbox?: [number, number, number, number];
          gesture?: 'waving' | 'hand_up' | 'checking' | 'none';
          gesture_conf?: number;
          confidence?: number;
        }>;
        timestamp?: number;
      }) => {
        if (!raw.camera_id) return;
        const dets = Array.isArray(raw.detections) ? raw.detections : [];
        let bestGesture: 'waving' | 'hand_up' | 'checking' | 'none' = 'none';
        let bestGestureConf = 0;
        const personDetections = dets.map((d) => {
          const g = (d.gesture ?? 'none') as 'waving' | 'hand_up' | 'checking' | 'none';
          const gc = d.gesture_conf ?? 0;
          if (g !== 'none' && gc > bestGestureConf) {
            bestGesture = g;
            bestGestureConf = gc;
          }
          return {
            bbox: (Array.isArray(d.bbox) && d.bbox.length === 4
              ? d.bbox : [0, 0, 0, 0]) as [number, number, number, number],
            confidence: d.confidence ?? 0,
            gesture: g,
            gesture_conf: gc,
          };
        });
        pendingDetectionsRef.current[raw.camera_id] = {
          camera_id: raw.camera_id,
          person_count:
            typeof raw.person_count === 'number' ? raw.person_count : dets.length,
          detections: personDetections,
          best_gesture: bestGesture,
          best_gesture_confidence: bestGestureConf,
          inference_ms: typeof raw.inference_ms === 'number' ? raw.inference_ms : 0,
          timestamp: raw.timestamp ?? Date.now(),
        };
      };

      ws.onmessage = (event) => {
        lastFrameTimeRef.current = Date.now();

        // 二进制帧 — [4B BE u32 header_len][header JSON][JPEG]
        // 不走 JSON.parse 大字符串、不走 base64 解码，主线程瓶颈消除
        if (event.data instanceof ArrayBuffer) {
          const buf = event.data;
          if (buf.byteLength < 4) return;
          const view = new DataView(buf);
          const headerLen = view.getUint32(0, false);
          if (headerLen <= 0 || 4 + headerLen > buf.byteLength) return;
          let header: { camera_id?: string } & Record<string, unknown>;
          try {
            const headerStr = new TextDecoder('utf-8').decode(
              new Uint8Array(buf, 4, headerLen)
            );
            header = JSON.parse(headerStr);
          } catch {
            return;
          }
          const cid = header.camera_id;
          if (!cid) return;
          const jpegBytes = new Uint8Array(buf, 4 + headerLen);
          const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
          const blobUrl = URL.createObjectURL(blob);
          // 帧到达速度 > rAF flush 节奏时，pending 槽位里的旧 URL 会被覆盖。
          // 不在这里 revoke 就成了泄漏（liveBlobUrlsRef 只追踪已注入 state 的那个）
          const stale = pendingFramesRef.current[cid];
          if (stale && stale.startsWith('blob:')) {
            try { URL.revokeObjectURL(stale); } catch { /* noop */ }
          }
          pendingFramesRef.current[cid] = blobUrl;
          // 采样端到端延迟：header.timestamp 单位秒
          // server_offset = server_clock - browser_clock；real_lag = raw_lag + offset
          const ts = (header as { timestamp?: number }).timestamp;
          if (typeof ts === 'number' && ts > 0) {
            const rawLagMs = Date.now() - ts * 1000;
            const lagMs = rawLagMs + serverOffsetMsRef.current;
            if (lagMs >= -500 && lagMs < 60_000) lagSamplesRef.current.push(lagMs);
          }
          const buckets = frameTimestampsRef.current;
          if (!buckets[cid]) buckets[cid] = [];
          buckets[cid].push(Date.now());
          ingestHeader(header as Parameters<typeof ingestHeader>[0]);
          scheduleFlush();
          return;
        }

        // 文本消息 — 控制信令（pong/status/subscribed 等）
        try {
          const msg: WebSocketMessage = JSON.parse(event.data);

          // 兼容旧 base64 帧（服务端如果回退到 send_json）
          if ('camera_id' in msg && 'frame' in msg) {
            const frameMsg = msg as unknown as FrameMessage & { frame?: string };
            const frameData = frameMsg.data || frameMsg.frame || '';
            const cid = frameMsg.camera_id;
            if (frameData) {
              pendingFramesRef.current[cid] = `data:image/jpeg;base64,${frameData}`;
              const buckets = frameTimestampsRef.current;
              if (!buckets[cid]) buckets[cid] = [];
              buckets[cid].push(Date.now());
            }
            ingestHeader(msg as unknown as Parameters<typeof ingestHeader>[0]);
            scheduleFlush();
            return;
          }

          // pong (诊断用) — 不在 WebSocketMessage 联合类型里，单独处理
          const maybePong = msg as { type?: string; client_time?: number; server_time?: number };
          if (maybePong.type === 'pong') {
            const ct = maybePong.client_time;
            const st = maybePong.server_time;
            if (typeof ct === 'number' && typeof st === 'number') {
              const nowSec = Date.now() / 1000;
              const rttMs = (nowSec - ct) * 1000;
              const offsetMs = (st - (ct + (nowSec - ct) / 2)) * 1000;
              // 落到 ref，让后续 [ws-lag] 自动减去时钟偏差，得到真实管道延迟
              serverOffsetMsRef.current = offsetMs;
              console.log(
                `[ws-clock] rtt=${rttMs.toFixed(1)}ms server_offset=${offsetMs.toFixed(1)}ms ` +
                `(>0=服务器时钟比浏览器快；[ws-lag] 已自动减去此偏差)`
              );
            }
            return;
          }

          switch (msg.type) {
            case 'status': {
              const statusMsg = msg as StatusMessage;
              if (statusMsg.fps) {
                setFps(statusMsg.fps);
              }
              break;
            }
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setConnecting(false);
        connectingRef.current = false;

        if (heartbeatTimerRef.current) {
          clearInterval(heartbeatTimerRef.current);
          heartbeatTimerRef.current = null;
        }

        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttemptsRef.current += 1;
          setLastError(
            `连接中断，正在重连 (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`
          );
          reconnectTimerRef.current = setTimeout(() => {
            connect();
          }, RECONNECT_INTERVAL);
        } else {
          setLastError('WebSocket 连接失败，请检查服务状态与网络。');
        }
      };

      ws.onerror = () => {
        setLastError('WebSocket 连接错误。');
        ws.close();
      };
    } catch {
      setConnecting(false);
      connectingRef.current = false;
      setLastError('初始化 WebSocket 失败。');
    }
  }, [getWebSocketUrl, scheduleFlush]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    // 释放残留 blob URL，包括待 flush 的与已注入 state 的
    for (const url of Object.values(pendingFramesRef.current)) {
      if (url && url.startsWith('blob:')) {
        try { URL.revokeObjectURL(url); } catch { /* noop */ }
      }
    }
    pendingFramesRef.current = {};
    for (const url of Object.values(liveBlobUrlsRef.current)) {
      if (url && url.startsWith('blob:')) {
        try { URL.revokeObjectURL(url); } catch { /* noop */ }
      }
    }
    liveBlobUrlsRef.current = {};
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(() => connect(), 100);
  }, [disconnect, connect]);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  useEffect(() => {
    const interval = setInterval(() => {
      updateFps();
    }, 500);
    return () => clearInterval(interval);
  }, [updateFps]);

  // 端到端延迟统计：每秒打一行 p50/p95/n 到 console，用来定位 3s 延迟瓶颈在哪段
  useEffect(() => {
    const interval = setInterval(() => {
      const samples = lagSamplesRef.current;
      if (samples.length === 0) return;
      lagSamplesRef.current = [];
      samples.sort((a, b) => a - b);
      const p = (q: number) => samples[Math.min(samples.length - 1, Math.floor(samples.length * q))];
      console.log(
        `[ws-lag] p50=${p(0.5)}ms p95=${p(0.95)}ms max=${samples[samples.length - 1]}ms n=${samples.length}`
      );
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (
        wsRef.current?.readyState === WebSocket.OPEN &&
        lastFrameTimeRef.current > 0 &&
        Date.now() - lastFrameTimeRef.current > FRAME_TIMEOUT_MS
      ) {
        console.warn(
          `[WebSocket] ${FRAME_TIMEOUT_MS}ms 未收到帧，主动断开重连`
        );
        reconnect();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [reconnect]);

  return {
    connected,
    connecting,
    lastError,
    frames,
    detections,
    fps,
    sendMessage,
    reconnect,
  };
}
