"""
DataLab SVG 图表生成器 — 零依赖矢量图表

为 Markdown/PPT 导出提供深色主题、可直接内嵌的 SVG 图表。
所有输出均为纯文本 XML，支持无限放大不失真。
"""

from typing import Dict, List, Any

ENGINE_COLORS = {
    "simple": "#94a3b8",
    "transformer": "#60a5fa",
    "triplelock": "#c084fc",
    "transformer_triplelock": "#fbbf24",
    "simple_transformer": "#34d399",
    "sth_full": "#34d399",
    "sth_no_softfilter": "#f87171",
    "sth_no_velocity_gate": "#fbbf24",
    "sth_no_pose_gate": "#c084fc",
    "sth_transformer_only": "#60a5fa",
    "simple_no_periodicity": "#fb923c",
    "simple_no_pose_gate": "#a78bfa",
    "triplelock_no_orientation": "#38bdf8",
    "simple_minicpm": "#f472b6",
    "minicpm": "#22d3ee",
    "simple_minicpm_full": "#f472b6",
    "simple_minicpm_no_simple_gate": "#fbbf24",
    "simple_minicpm_no_cooldown": "#c084fc",
    "minicpm_full": "#22d3ee",
    "simple_full": "#94a3b8",
}

ENGINE_LABELS_SHORT = {
    "simple": "Simple",
    "transformer": "TF",
    "triplelock": "TL",
    "transformer_triplelock": "TF+TL",
    "simple_transformer": "STH",
    "sth_full": "STH",
    "sth_no_softfilter": "-soft",
    "sth_no_velocity_gate": "-vel",
    "sth_no_pose_gate": "-pose",
    "sth_transformer_only": "TFonly",
    "simple_no_periodicity": "-period",
    "simple_no_pose_gate": "-pose",
    "triplelock_no_orientation": "-orient",
    "simple_minicpm": "S+M",
    "minicpm": "MCPM",
    "simple_minicpm_full": "S+M",
    "simple_minicpm_no_simple_gate": "-gate",
    "simple_minicpm_no_cooldown": "-cd",
    "minicpm_full": "MCPM",
    "simple_full": "Simple",
}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _font(size: int = 10, color: str = "#94a3b8", weight: str = "normal") -> str:
    return (
        f'font-size="{size}" font-family="Noto Sans CJK SC,Noto Sans SC,WenQuanYi Zen Hei,sans-serif" '
        f'fill="{color}" font-weight="{weight}"'
    )


def svg_bar_chart(
    data: List[Dict[str, Any]],
    title: str,
    value_fmt: str = "{:.1f}",
    width: int = 800,
    height: int = 360,
) -> str:
    """深色主题柱状图 SVG。"""
    m = {"t": 50, "r": 40, "b": 90, "l": 80}
    cw = width - m["l"] - m["r"]
    ch = height - m["t"] - m["b"]

    max_val = max(d["value"] for d in data) if data else 1
    max_val = max(max_val, 0.001)
    n = len(data)
    gap_ratio = 0.35
    bw = cw / (n * (1 + gap_ratio)) if n > 0 else 0
    gap = bw * gap_ratio

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="32" text-anchor="middle" {_font(15, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    # 网格 + Y轴
    for i in range(6):
        y = m["t"] + ch * i / 5
        val = max_val * (1 - i / 5)
        p.append(
            f'<line x1="{m["l"]}" y1="{y}" x2="{width - m["r"]}" y2="{y}" stroke="#1e293b" stroke-width="1"/>'
        )
        p.append(f'<text x="{m["l"] - 10}" y="{y + 4}" text-anchor="end" {_font(10, "#64748b")}>{value_fmt.format(val)}</text>')

    # 柱条
    for i, d in enumerate(data):
        x = m["l"] + i * (bw + gap) + gap / 2
        h = (d["value"] / max_val) * ch
        y = m["t"] + ch - h
        color = d.get("color", "#38bdf8")
        p.append(f'<rect x="{x}" y="{y}" width="{bw * 0.9}" height="{h}" fill="{color}" rx="5" opacity="0.92"/>')
        p.append(
            f'<text x="{x + bw * 0.45}" y="{y - 6}" text-anchor="middle" {_font(10, "#e2e8f0")}>'
            f'{value_fmt.format(d["value"])}</text>'
        )
        label = _esc(d["label"])
        lx = x + bw * 0.45
        ly = height - m["b"] + 18
        p.append(
            f'<text x="{lx}" y="{ly}" text-anchor="end" {_font(10)} transform="rotate(-35, {lx}, {ly})">'
            f'{label}</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def svg_line_chart(
    data: List[Dict[str, Any]],
    title: str,
    x_key: str = "x",
    y_keys: List[str] = None,
    y_labels: Dict[str, str] = None,
    y_colors: Dict[str, str] = None,
    width: int = 800,
    height: int = 360,
) -> str:
    """深色主题折线图 SVG。"""
    m = {"t": 50, "r": 150, "b": 60, "l": 80}
    cw = width - m["l"] - m["r"]
    ch = height - m["t"] - m["b"]

    if not data or not y_keys:
        return f'<svg viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#0f172a" rx="12"/></svg>'

    all_y = [d[k] for d in data for k in y_keys]
    min_y = min(all_y)
    max_y = max(all_y)
    if max_y == min_y:
        max_y = min_y + 1

    min_x = min(d[x_key] for d in data)
    max_x = max(d[x_key] for d in data)
    if max_x == min_x:
        max_x = min_x + 1

    def tx(x: float) -> float:
        return m["l"] + (x - min_x) / (max_x - min_x) * cw

    def ty(y: float) -> float:
        return m["t"] + ch - (y - min_y) / (max_y - min_y) * ch

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="32" text-anchor="middle" {_font(15, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    # 水平网格
    for i in range(6):
        y = m["t"] + ch * i / 5
        val = min_y + (max_y - min_y) * (1 - i / 5)
        p.append(
            f'<line x1="{m["l"]}" y1="{y}" x2="{width - m["r"]}" y2="{y}" stroke="#1e293b" stroke-width="1"/>'
        )
        p.append(f'<text x="{m["l"] - 10}" y="{y + 4}" text-anchor="end" {_font(10, "#64748b")}>{val:.1f}</text>')

    # 垂直网格
    for i in range(6):
        x = m["l"] + cw * i / 5
        val = min_x + (max_x - min_x) * i / 5
        p.append(
            f'<line x1="{x}" y1="{m["t"]}" x2="{x}" y2="{height - m["b"]}" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        p.append(f'<text x="{x}" y="{height - m["b"] + 18}" text-anchor="middle" {_font(10, "#64748b")}>{val:.2f}</text>')

    # 折线 + 点
    for k in y_keys:
        pts = " ".join(f"{tx(d[x_key])},{ty(d[k])}" for d in data)
        color = (y_colors or {}).get(k, "#38bdf8")
        p.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for d in data:
            p.append(
                f'<circle cx="{tx(d[x_key])}" cy="{ty(d[k])}" r="3.5" fill="{color}" stroke="#0f172a" stroke-width="1.5"/>'
            )

    # 图例
    for idx, k in enumerate(y_keys):
        ly = m["t"] + 20 + idx * 24
        lx = width - m["r"] + 20
        color = (y_colors or {}).get(k, "#38bdf8")
        label = (y_labels or {}).get(k, k)
        p.append(f'<rect x="{lx}" y="{ly - 8}" width="16" height="3" rx="1.5" fill="{color}"/>')
        p.append(f'<text x="{lx + 22}" y="{ly}" {_font(11)}>{_esc(label)}</text>')

    p.append("</svg>")
    return "\n".join(p)


def svg_xy_lines_chart(
    series_data: Dict[str, List[tuple]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    series_colors: Dict[str, str] = None,
    series_labels: Dict[str, str] = None,
    width: int = 600,
    height: int = 520,
    show_diagonal: bool = False,
) -> str:
    """多系列 XY 折线图（PR / ROC 曲线专用）。

    series_data: {series_name: [(x, y), ...]}
    """
    import math

    m = {"t": 55, "r": 170, "b": 70, "l": 80}
    cw = width - m["l"] - m["r"]
    ch = height - m["t"] - m["b"]

    if not series_data:
        return f'<svg viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#0f172a" rx="12"/></svg>'

    all_x = [x for pts in series_data.values() for x, _ in pts]
    all_y = [y for pts in series_data.values() for _, y in pts]
    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 1.0
    min_y = min(all_y) if all_y else 0.0
    max_y = max(all_y) if all_y else 1.0

    # 动态坐标轴 + padding，但限制最小范围避免过度压缩
    pad_x = (max_x - min_x) * 0.1 if max_x > min_x else 0.05
    pad_y = (max_y - min_y) * 0.1 if max_y > min_y else 0.05
    min_x = max(0.0, min_x - pad_x)
    max_x = min(1.0, max_x + pad_x)
    min_y = max(0.0, min_y - pad_y)
    max_y = min(1.0, max_y + pad_y)

    # 强制最小显示范围，防止单点或极小范围导致视觉失真
    min_range = 0.15
    if max_x - min_x < min_range:
        cx = (min_x + max_x) / 2
        min_x = max(0.0, cx - min_range / 2)
        max_x = min(1.0, cx + min_range / 2)
    if max_y - min_y < min_range:
        cy = (min_y + max_y) / 2
        min_y = max(0.0, cy - min_range / 2)
        max_y = min(1.0, cy + min_range / 2)

    if max_x == min_x:
        max_x = min_x + 0.01
    if max_y == min_y:
        max_y = min_y + 0.01

    def tx(x: float) -> float:
        return m["l"] + (x - min_x) / (max_x - min_x) * cw

    def ty(y: float) -> float:
        return m["t"] + ch - (y - min_y) / (max_y - min_y) * ch

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="32" text-anchor="middle" {_font(15, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    # 网格
    for i in range(6):
        y = m["t"] + ch * i / 5
        val = min_y + (max_y - min_y) * (1 - i / 5)
        p.append(
            f'<line x1="{m["l"]}" y1="{y}" x2="{width - m["r"]}" y2="{y}" stroke="#1e293b" stroke-width="1"/>'
        )
        p.append(f'<text x="{m["l"] - 10}" y="{y + 4}" text-anchor="end" {_font(10, "#64748b")}>{val:.2f}</text>')

    for i in range(6):
        x = m["l"] + cw * i / 5
        val = min_x + (max_x - min_x) * i / 5
        p.append(
            f'<line x1="{x}" y1="{m["t"]}" x2="{x}" y2="{height - m["b"]}" stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        p.append(f'<text x="{x}" y="{height - m["b"] + 18}" text-anchor="middle" {_font(10, "#64748b")}>{val:.2f}</text>')

    # 轴标签
    if x_label:
        p.append(
            f'<text x="{m["l"] + cw / 2}" y="{height - 18}" text-anchor="middle" {_font(12, "#94a3b8")}>'
            f'{_esc(x_label)}</text>'
        )
    if y_label:
        p.append(
            f'<text x="20" y="{m["t"] + ch / 2}" text-anchor="middle" transform="rotate(-90, 20, {m["t"] + ch / 2})" '
            f'{_font(12, "#94a3b8")}>{_esc(y_label)}</text>'
        )

    # 对角参考线（ROC 随机猜测基线）
    if show_diagonal:
        p.append(
            f'<line x1="{tx(0)}" y1="{ty(0)}" x2="{tx(1)}" y2="{ty(1)}" '
            f'stroke="#64748b" stroke-width="1.5" stroke-dasharray="6,4"/>'
        )

    # 折线 + 点（按 x 排序）
    for s_idx, (s_name, pts) in enumerate(series_data.items()):
        if not pts:
            continue
        sorted_pts = sorted(pts, key=lambda xy: xy[0])
        color = (series_colors or {}).get(s_name, "#38bdf8")
        pts_str = " ".join(f"{tx(x)},{ty(y)}" for x, y in sorted_pts)
        p.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for x, y in sorted_pts:
            p.append(
                f'<circle cx="{tx(x)}" cy="{ty(y)}" r="3" fill="{color}" stroke="#0f172a" stroke-width="1.5"/>'
            )

    # 图例
    for idx, s_name in enumerate(series_data.keys()):
        ly = m["t"] + idx * 22
        lx = width - m["r"] + 20
        color = (series_colors or {}).get(s_name, "#38bdf8")
        label = (series_labels or {}).get(s_name, s_name)
        p.append(f'<rect x="{lx}" y="{ly - 6}" width="14" height="3" rx="1.5" fill="{color}"/>')
        p.append(f'<text x="{lx + 20}" y="{ly}" {_font(11)}>{_esc(label)}</text>')

    p.append("</svg>")
    return "\n".join(p)


def svg_heatmap(
    matrix: Dict[str, Dict[str, float]],
    labels: List[str],
    title: str,
    width: int = 520,
    height: int = 520,
) -> str:
    """一致率矩阵热力图 SVG。"""
    n = len(labels)
    cell = min((width - 160) / n, (height - 160) / n)
    ox = (width - cell * n) / 2 + 60
    oy = (height - cell * n) / 2 + 60

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="32" text-anchor="middle" {_font(15, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            val = matrix.get(row, {}).get(col, 0)
            is_diag = i == j
            if is_diag:
                fill = "rgba(16, 185, 129, 0.18)"
                tc = "#34d399"
            else:
                intensity = val * 0.45
                fill = f"rgba(56, 189, 248, {intensity})"
                tc = "#e2e8f0" if val > 0.5 else "#94a3b8"
            x = ox + j * cell
            y = oy + i * cell
            p.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#1e293b" stroke-width="1"/>')
            p.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" {_font(11, tc)}>'
                f'{(val * 100):.0f}%</text>'
            )

    for j, label in enumerate(labels):
        short = ENGINE_LABELS_SHORT.get(label, label[:6])
        p.append(
            f'<text x="{ox + j * cell + cell / 2}" y="{oy - 8}" text-anchor="middle" {_font(10)}>'
            f'{_esc(short)}</text>'
        )

    for i, label in enumerate(labels):
        short = ENGINE_LABELS_SHORT.get(label, label[:6])
        p.append(
            f'<text x="{ox - 8}" y="{oy + i * cell + cell / 2 + 4}" text-anchor="end" {_font(10)}>'
            f'{_esc(short)}</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def svg_grouped_bar_chart(
    data: List[Dict[str, Any]],
    group_key: str = "name",
    series_keys: List[str] = None,
    series_colors: Dict[str, str] = None,
    series_labels: Dict[str, str] = None,
    title: str = "",
    width: int = 800,
    height: int = 360,
) -> str:
    """分组柱状图 SVG（场景分析）。"""
    m = {"t": 50, "r": 40, "b": 90, "l": 80}
    cw = width - m["l"] - m["r"]
    ch = height - m["t"] - m["b"]

    if not data or not series_keys:
        return f'<svg viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#0f172a" rx="12"/></svg>'

    all_vals = [d.get(k, 0) for d in data for k in series_keys]
    max_val = max(all_vals) if all_vals else 1
    max_val = max(max_val, 0.001)

    n_groups = len(data)
    n_series = len(series_keys)
    group_gap = min(30, cw / (n_groups * 3)) if n_groups > 0 else 0
    bar_w = (cw - group_gap * (n_groups + 1)) / (n_groups * n_series)
    if bar_w < 10:
        bar_w = 10

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="32" text-anchor="middle" {_font(15, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    for i in range(6):
        y = m["t"] + ch * i / 5
        val = max_val * (1 - i / 5)
        p.append(
            f'<line x1="{m["l"]}" y1="{y}" x2="{width - m["r"]}" y2="{y}" stroke="#1e293b" stroke-width="1"/>'
        )
        p.append(f'<text x="{m["l"] - 10}" y="{y + 4}" text-anchor="end" {_font(10, "#64748b")}>{val:.0f}</text>')

    for g_idx, d in enumerate(data):
        gx = m["l"] + group_gap + g_idx * (n_series * bar_w + group_gap)
        for s_idx, k in enumerate(series_keys):
            val = d.get(k, 0)
            x = gx + s_idx * bar_w
            h = (val / max_val) * ch
            y = m["t"] + ch - h
            color = (series_colors or {}).get(k, "#38bdf8")
            p.append(f'<rect x="{x}" y="{y}" width="{bar_w * 0.85}" height="{h}" fill="{color}" rx="3"/>')

        lx = gx + (n_series * bar_w) / 2
        ly = height - m["b"] + 20
        p.append(
            f'<text x="{lx}" y="{ly}" text-anchor="end" {_font(10)} transform="rotate(-25, {lx}, {ly})">'
            f'{_esc(d.get(group_key, ""))}</text>'
        )

    # 图例放右上角
    for idx, k in enumerate(series_keys):
        ly = m["t"] + idx * 18
        lx = width - m["r"] + 10
        color = (series_colors or {}).get(k, "#38bdf8")
        label = (series_labels or {}).get(k, k)
        p.append(f'<rect x="{lx}" y="{ly - 6}" width="12" height="3" rx="1.5" fill="{color}"/>')
        p.append(f'<text x="{lx + 16}" y="{ly}" {_font(10)}>{_esc(label)}</text>')

    p.append("</svg>")
    return "\n".join(p)


def svg_radar_chart(
    data: List[Dict[str, Any]],
    dimensions: List[str],
    series_keys: List[str],
    series_colors: Dict[str, str] = None,
    series_labels: Dict[str, str] = None,
    title: str = "",
    width: int = 520,
    height: int = 420,
) -> str:
    """雷达图 SVG（深色主题）。"""
    cx = width / 2
    cy = height / 2 + 10
    r = min(cx, cy) - 70
    n = len(dimensions)
    if n == 0 or not series_keys:
        return f'<svg viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#0f172a" rx="12"/></svg>'

    # 预计算维度角度（从顶部开始，顺时针）
    angles = [i * 2 * 3.141592653589793 / n - 3.141592653589793 / 2 for i in range(n)]

    def px(val: float, idx: int) -> float:
        return cx + val * r * math.cos(angles[idx])

    def py(val: float, idx: int) -> float:
        return cy + val * r * math.sin(angles[idx])

    import math

    p: List[str] = []
    p.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>')
    p.append(
        f'<text x="{width / 2}" y="28" text-anchor="middle" {_font(14, "#e2e8f0", "600")}>'
        f'{_esc(title)}</text>'
    )

    # 网格多边形
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = " ".join(f"{px(level, i)},{py(level, i)}" for i in range(n))
        p.append(
            f'<polygon points="{pts}" fill="none" stroke="#1e293b" stroke-width="1"/>'
        )

    # 轴线
    for i in range(n):
        p.append(
            f'<line x1="{cx}" y1="{cy}" x2="{px(1.0, i)}" y2="{py(1.0, i)}" stroke="#1e293b" stroke-width="1"/>'
        )

    # 维度标签
    for i, dim in enumerate(dimensions):
        lx = px(1.15, i)
        ly = py(1.15, i)
        anchor = "middle"
        if lx < cx - 5:
            anchor = "end"
        elif lx > cx + 5:
            anchor = "start"
        p.append(
            f'<text x="{lx}" y="{ly + 4}" text-anchor="{anchor}" {_font(11, "#94a3b8")}>{_esc(dim)}</text>'
        )

    # 数据多边形
    for skey in series_keys:
        color = (series_colors or {}).get(skey, "#38bdf8")
        pts = " ".join(
            f"{px(data[i].get(skey, 0), i)},{py(data[i].get(skey, 0), i)}" for i in range(n)
        )
        p.append(
            f'<polygon points="{pts}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        for i in range(n):
            p.append(
                f'<circle cx="{px(data[i].get(skey, 0), i)}" cy="{py(data[i].get(skey, 0), i)}" r="3" fill="{color}" stroke="#0f172a" stroke-width="1"/>'
            )

    # 图例（右下角）
    for idx, skey in enumerate(series_keys):
        ly = height - 30 - (len(series_keys) - 1 - idx) * 18
        lx = width - 20
        color = (series_colors or {}).get(skey, "#38bdf8")
        label = (series_labels or {}).get(skey, skey)
        p.append(f'<rect x="{lx - 12}" y="{ly - 6}" width="12" height="3" rx="1.5" fill="{color}"/>')
        p.append(f'<text x="{lx - 16}" y="{ly}" text-anchor="end" {_font(10)}>{_esc(label)}</text>')

    p.append("</svg>")
    return "\n".join(p)


def svg_to_png(svg_string: str, scale: float = 2.0) -> bytes:
    """将 SVG 字符串转换为 PNG 字节（依赖 cairosvg）。"""
    try:
        import cairosvg
    except ImportError:
        raise RuntimeError("cairosvg 未安装，无法导出 PNG")
    return cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), scale=scale)
