#!/usr/bin/env python3
"""
Usage:
    python3 tools/generate_ieee_style_figures.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

FONT_FAMILY = "Times New Roman, Times, serif"
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12
ANNOTATION_SIZE = 11

COLOR_BLUE = "#1f4e79"
COLOR_ORANGE = "#c55a11"
COLOR_GREEN = "#5b8c5a"
COLOR_GRAY = "#7f7f7f"
COLOR_GRID = "#d9d9d9"
COLOR_AXIS = "#000000"
COLOR_BG = "#ffffff"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_number(text: str) -> float:
    cleaned = (
        text.replace("`", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )
    return float(cleaned)


def parse_markdown_table(text: str, heading: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    start = None

    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx + 1
            break

    if start is None:
        raise ValueError(f"Heading not found: {heading}")

    while start < len(lines) and not lines[start].strip().startswith("|"):
        start += 1

    if start + 1 >= len(lines):
        raise ValueError(f"No markdown table after heading: {heading}")

    header_cells = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    data_rows: List[Dict[str, str]] = []

    idx = start + 2
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header_cells):
            data_rows.append(dict(zip(header_cells, cells)))
        idx += 1

    return data_rows


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLOR_BG}"/>',
    ]


def svg_footer(lines: List[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def svg_text(lines: List[str], x: float, y: float, text: str, size: int, anchor: str = "middle",
             weight: str = "normal", fill: str = COLOR_AXIS, rotate: float | None = None) -> None:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    safe_text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    lines.append(
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{safe_text}</text>'
    )


def draw_axes(lines: List[str], x0: float, y0: float, x1: float, y1: float) -> None:
    lines.append(
        f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
        f'stroke="{COLOR_AXIS}" stroke-width="1.4"/>'
    )
    lines.append(
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" '
        f'stroke="{COLOR_AXIS}" stroke-width="1.4"/>'
    )


def draw_y_grid(lines: List[str], x0: float, x1: float, y0: float, y1: float,
                y_ticks: Sequence[float], y_min: float, y_max: float) -> None:
    for tick in y_ticks:
        ratio = 0.0 if y_max == y_min else (tick - y_min) / (y_max - y_min)
        y = y1 - ratio * (y1 - y0)
        lines.append(
            f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" '
            f'stroke="{COLOR_GRID}" stroke-width="0.8"/>'
        )


def nice_ticks(y_min: float, y_max: float, tick_count: int = 5) -> List[float]:
    if y_max <= y_min:
        return [y_min]

    raw_step = (y_max - y_min) / tick_count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude

    if residual < 1.5:
        nice_step = 1 * magnitude
    elif residual < 3:
        nice_step = 2 * magnitude
    elif residual < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(y_min / nice_step) * nice_step
    end = math.ceil(y_max / nice_step) * nice_step

    ticks = []
    value = start
    while value <= end + 1e-12:
        ticks.append(round(value, 10))
        value += nice_step
    return ticks


def format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value/1000:.0f}k"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def make_single_bar_chart(
    title: str,
    categories: Sequence[str],
    values: Sequence[float],
    y_label: str,
    output_path: Path,
    color: str = COLOR_BLUE,
) -> None:
    width, height = 760, 460
    left, right, top, bottom = 90, 30, 60, 75
    plot_x0, plot_y0 = left, top
    plot_x1, plot_y1 = width - right, height - bottom

    ymax = max(values) * 1.18
    ymin = 0.0
    ticks = nice_ticks(ymin, ymax, 5)

    lines = svg_header(width, height)
    svg_text(lines, width / 2, 28, title, TITLE_SIZE, weight="bold")
    draw_y_grid(lines, plot_x0, plot_x1, plot_y0, plot_y1, ticks, ymin, ticks[-1])
    draw_axes(lines, plot_x0, plot_y0, plot_x1, plot_y1)

    svg_text(lines, width / 2, height - 20, "Implementation", LABEL_SIZE)
    svg_text(lines, 22, height / 2, y_label, LABEL_SIZE, rotate=-90)

    for tick in ticks:
        ratio = 0.0 if ticks[-1] == ymin else (tick - ymin) / (ticks[-1] - ymin)
        y = plot_y1 - ratio * (plot_y1 - plot_y0)
        lines.append(
            f'<line x1="{plot_x0 - 5:.2f}" y1="{y:.2f}" x2="{plot_x0:.2f}" y2="{y:.2f}" '
            f'stroke="{COLOR_AXIS}" stroke-width="1"/>'
        )
        svg_text(lines, plot_x0 - 10, y + 4, format_tick(tick), TICK_SIZE, anchor="end")

    count = len(categories)
    slot = (plot_x1 - plot_x0) / count
    bar_w = slot * 0.5

    for idx, (cat, value) in enumerate(zip(categories, values)):
        cx = plot_x0 + slot * (idx + 0.5)
        x = cx - bar_w / 2
        h = 0.0 if ticks[-1] == ymin else (value - ymin) / (ticks[-1] - ymin) * (plot_y1 - plot_y0)
        y = plot_y1 - h
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
            f'fill="{color}" stroke="{COLOR_AXIS}" stroke-width="0.8"/>'
        )
        svg_text(lines, cx, plot_y1 + 24, cat, TICK_SIZE)
        svg_text(lines, cx, y - 6, format_tick(value), ANNOTATION_SIZE)

    output_path.write_text(svg_footer(lines), encoding="utf-8")


def make_dual_panel_bar_chart(
    title: str,
    left_title: str,
    left_categories: Sequence[str],
    left_values: Sequence[float],
    right_title: str,
    right_categories: Sequence[str],
    right_values: Sequence[float],
    y_label: str,
    output_path: Path,
) -> None:
    width, height = 980, 430
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 55, 65
    panel_gap = 45
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = height - margin_top - margin_bottom

    lines = svg_header(width, height)
    svg_text(lines, width / 2, 28, title, TITLE_SIZE, weight="bold")
    svg_text(lines, 22, height / 2, y_label, LABEL_SIZE, rotate=-90)

    def draw_panel(x0: float, panel_title: str, cats: Sequence[str], vals: Sequence[float], color: str) -> None:
        x1 = x0 + panel_width
        y0 = margin_top
        y1 = margin_top + panel_height
        ymin = 0.0
        ymax = max(vals) * 1.22
        ticks = nice_ticks(ymin, ymax, 4)

        svg_text(lines, (x0 + x1) / 2, y0 - 12, panel_title, LABEL_SIZE, weight="bold")
        draw_y_grid(lines, x0, x1, y0, y1, ticks, ymin, ticks[-1])
        draw_axes(lines, x0, y0, x1, y1)

        for tick in ticks:
            ratio = 0.0 if ticks[-1] == ymin else (tick - ymin) / (ticks[-1] - ymin)
            y = y1 - ratio * (y1 - y0)
            lines.append(
                f'<line x1="{x0 - 5:.2f}" y1="{y:.2f}" x2="{x0:.2f}" y2="{y:.2f}" '
                f'stroke="{COLOR_AXIS}" stroke-width="1"/>'
            )
            svg_text(lines, x0 - 8, y + 4, format_tick(tick), TICK_SIZE, anchor="end")

        slot = (x1 - x0) / len(cats)
        bar_w = slot * 0.48
        for idx, (cat, value) in enumerate(zip(cats, vals)):
            cx = x0 + slot * (idx + 0.5)
            x = cx - bar_w / 2
            h = 0.0 if ticks[-1] == ymin else (value - ymin) / (ticks[-1] - ymin) * (y1 - y0)
            y = y1 - h
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                f'fill="{color}" stroke="{COLOR_AXIS}" stroke-width="0.8"/>'
            )
            svg_text(lines, cx, y1 + 22, cat, TICK_SIZE)
            svg_text(lines, cx, y - 6, format_tick(value), ANNOTATION_SIZE)

    draw_panel(margin_left, left_title, left_categories, left_values, COLOR_BLUE)
    draw_panel(margin_left + panel_width + panel_gap, right_title, right_categories, right_values, COLOR_ORANGE)

    output_path.write_text(svg_footer(lines), encoding="utf-8")


def make_dual_panel_line_chart(
    title: str,
    left_title: str,
    xs: Sequence[float],
    left_ys: Sequence[float],
    left_y_label: str,
    right_title: str,
    right_ys: Sequence[float],
    right_y_label: str,
    output_path: Path,
) -> None:
    width, height = 980, 430
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 55, 70
    panel_gap = 45
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = height - margin_top - margin_bottom

    lines = svg_header(width, height)
    svg_text(lines, width / 2, 28, title, TITLE_SIZE, weight="bold")

    def draw_panel(x0: float, panel_title: str, ys: Sequence[float], y_label: str, color: str) -> None:
        x1 = x0 + panel_width
        y0 = margin_top
        y1 = margin_top + panel_height
        xmin, xmax = min(xs), max(xs)
        ymin = 0.0
        ymax = max(ys) * 1.18
        ticks = nice_ticks(ymin, ymax, 4)

        svg_text(lines, (x0 + x1) / 2, y0 - 12, panel_title, LABEL_SIZE, weight="bold")
        svg_text(lines, (x0 + x1) / 2, height - 18, "Number of Patterns", LABEL_SIZE)
        svg_text(lines, x0 - 46, (y0 + y1) / 2, y_label, LABEL_SIZE, rotate=-90)
        draw_y_grid(lines, x0, x1, y0, y1, ticks, ymin, ticks[-1])
        draw_axes(lines, x0, y0, x1, y1)

        for tick in ticks:
            ratio = 0.0 if ticks[-1] == ymin else (tick - ymin) / (ticks[-1] - ymin)
            y = y1 - ratio * (y1 - y0)
            lines.append(
                f'<line x1="{x0 - 5:.2f}" y1="{y:.2f}" x2="{x0:.2f}" y2="{y:.2f}" '
                f'stroke="{COLOR_AXIS}" stroke-width="1"/>'
            )
            svg_text(lines, x0 - 8, y + 4, format_tick(tick), TICK_SIZE, anchor="end")

        points: List[Tuple[float, float]] = []
        for value_x, value_y in zip(xs, ys):
            xr = 0.0 if xmax == xmin else (value_x - xmin) / (xmax - xmin)
            yr = 0.0 if ticks[-1] == ymin else (value_y - ymin) / (ticks[-1] - ymin)
            px = x0 + xr * (x1 - x0)
            py = y1 - yr * (y1 - y0)
            points.append((px, py))
            svg_text(lines, px, y1 + 22, str(int(value_x)), TICK_SIZE)
            lines.append(
                f'<line x1="{px:.2f}" y1="{y1:.2f}" x2="{px:.2f}" y2="{y1 + 4:.2f}" '
                f'stroke="{COLOR_AXIS}" stroke-width="1"/>'
            )

        polyline = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{polyline}"/>'
        )
        for (px, py), value_y in zip(points, ys):
            lines.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.0" fill="{color}" stroke="{COLOR_AXIS}" stroke-width="0.7"/>'
            )
            svg_text(lines, px, py - 8, format_tick(value_y), ANNOTATION_SIZE)

    draw_panel(margin_left, left_title, left_ys, left_y_label, COLOR_BLUE)
    draw_panel(margin_left + panel_width + panel_gap, right_title, right_ys, right_y_label, COLOR_ORANGE)
    output_path.write_text(svg_footer(lines), encoding="utf-8")


def parse_part123_metrics() -> Tuple[List[str], List[float], List[float], List[float]]:
    text = read_text(ROOT / "Overall_Results_Analysis.md")
    rows = parse_markdown_table(text, "## 正式比較表")

    sim_seconds = []
    num_cycles = []
    dcache_miss = []
    parts = ["Part 1", "Part 2", "Part 3"]
    part_keys = ["Part 1 Scalar", "Part 2 RVV", "Part 3 SIMD-like RVV"]

    for row in rows:
        metric = row["Metric"].replace("`", "")
        if metric == "simSeconds":
            sim_seconds = [clean_number(row[key]) for key in part_keys]
        if metric == "numCycles":
            num_cycles = [clean_number(row[key]) for key in part_keys]
        if metric == "D-cache miss rate":
            dcache_miss = [clean_number(row[key]) for key in part_keys]

    return parts, sim_seconds, num_cycles, dcache_miss


def parse_part4_sweeps() -> Tuple[List[str], List[float], List[str], List[float]]:
    text = read_text(ROOT / "part4" / "results" / "Part4_Experiment_Summary.md")
    rows_ls = parse_markdown_table(text, "## TPB Sweep for LS Shared Kernel")
    rows_eq = parse_markdown_table(text, "## TPB Sweep for LMMSE Kernel")

    ls_labels = [f"TPB {row['TPB_LS']}" for row in rows_ls]
    ls_vals = [clean_number(row["Pipeline ms"]) for row in rows_ls]
    eq_labels = [f"TPB {row['TPB_EQ']}" for row in rows_eq]
    eq_vals = [clean_number(row["Pipeline ms"]) for row in rows_eq]
    return ls_labels, ls_vals, eq_labels, eq_vals


def parse_part4_shared_vs_serial() -> Tuple[List[str], List[float], List[float]]:
    text = read_text(ROOT / "part4" / "results" / "Part4_Experiment_Summary.md")
    rows = parse_markdown_table(text, "## Shared vs Serial LS Comparison")

    labels = []
    ls_vals = []
    pipeline_vals = []

    for row in rows:
        labels.append(row["LS Mode"].capitalize())
        ls_vals.append(clean_number(row["LS ms"]))
        pipeline_vals.append(clean_number(row["Pipeline ms"]))

    return labels, ls_vals, pipeline_vals


def parse_part5_sweep() -> Tuple[List[int], List[float], List[float]]:
    text = read_text(ROOT / "part5" / "results" / "Part5_Experiment_Summary.md")
    rows = parse_markdown_table(text, "## Pattern Sweep")
    patterns = [int(row["Patterns"]) for row in rows]
    pipeline = [clean_number(row["Pipeline ms"]) for row in rows]
    per_pattern = [p_ms / p for p_ms, p in zip(pipeline, patterns)]
    return patterns, pipeline, per_pattern


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    parts, sim_seconds, num_cycles, dcache_miss = parse_part123_metrics()
    make_single_bar_chart(
        title="gem5 SimSeconds Comparison for Parts 1-3",
        categories=parts,
        values=sim_seconds,
        y_label="Simulated Time (s)",
        output_path=FIG_DIR / "fig_part123_simseconds.svg",
        color=COLOR_GREEN,
    )

    make_single_bar_chart(
        title="gem5 NumCycles Comparison for Parts 1-3",
        categories=parts,
        values=num_cycles,
        y_label="Number of Cycles",
        output_path=FIG_DIR / "fig_part123_numcycles.svg",
        color=COLOR_BLUE,
    )

    make_single_bar_chart(
        title="gem5 D-Cache Miss Rate for Parts 1-3",
        categories=parts,
        values=dcache_miss,
        y_label="Miss Rate",
        output_path=FIG_DIR / "fig_part123_dcache_miss.svg",
        color=COLOR_ORANGE,
    )

    ls_labels, ls_vals, eq_labels, eq_vals = parse_part4_sweeps()
    make_dual_panel_bar_chart(
        title="Part 4 TPB Sweep (CUDA Kernel-Only Timing)",
        left_title="Stage 1 Sweep",
        left_categories=ls_labels,
        left_values=ls_vals,
        right_title="Stage 2 Sweep",
        right_categories=eq_labels,
        right_values=eq_vals,
        y_label="Pipeline Kernel Time (ms)",
        output_path=FIG_DIR / "fig_part4_tpb_sweep.svg",
    )

    shared_labels, shared_ls_vals, shared_pipeline_vals = parse_part4_shared_vs_serial()
    make_dual_panel_bar_chart(
        title="Part 4 Shared vs Serial Stage 1",
        left_title="Stage 1 LS Kernel Time",
        left_categories=shared_labels,
        left_values=shared_ls_vals,
        right_title="End-to-End Pipeline Kernel Time",
        right_categories=shared_labels,
        right_values=shared_pipeline_vals,
        y_label="Kernel Time (ms)",
        output_path=FIG_DIR / "fig_part4_shared_vs_serial.svg",
    )

    patterns, pipeline, per_pattern = parse_part5_sweep()
    make_dual_panel_line_chart(
        title="Part 5 Pattern Scaling (CUDA Kernel-Only Timing)",
        left_title="Total Pipeline Time",
        xs=patterns,
        left_ys=pipeline,
        left_y_label="Pipeline Time (ms)",
        right_title="Pipeline Time Per Pattern",
        right_ys=per_pattern,
        right_y_label="Time / Pattern (ms)",
        output_path=FIG_DIR / "fig_part5_pattern_scaling.svg",
    )

    print(f"Wrote figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("*.svg")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
